import numpy as np
from typing import List
from agent1 import Agent as Agent
from ReplayBuffer import ReplayBuffer
from env import Env
import tensorflow as tf
import configparser
import random
import time
from keras.models import Sequential, Model
#import matplotlib.pyplot as plt
import math
import os

Config = configparser.ConfigParser()
Config.read("./config.ini")

def list_add(re, des):
    for i in range(len(re)):
        des[i] += re[i]
    return des

def get_type(sellers : List[Agent], buyers : List[Agent]):
    s_type = [0] * len(sellers)
    b_type = [0] * len(buyers)
    for i in range(len(sellers)):
        s_type[i] = sellers[i].type
    for j in range(len(buyers)):
        b_type[j] = buyers[j].type
    return s_type, b_type


def avr_reward(ep_reward : list):
    max_episode_step = int(Config.get("Common", "Max_EP_Steps"))
    for i in range(len(ep_reward)):
        ep_reward[i] = ep_reward[i] / max_episode_step
    return ep_reward

#计算积分
def integral(cumulate_prob : list, isLess : bool, pricing, bottom, up): # 函数(定值) 上下限
    '''
    :param prob: 概率分布函数
    :param pricing:  此刻一轮的市场定价
    :param bottom: 积分下限
    :param up:  积分上限
    :return:  积分结果
    '''
    if bottom is None or up is None:
        return None
    else:
        up_index = int(np.round(up, 2) * 100 - 1) if up < 1 and up > 0 else 0 if up == 0 else 99
        bottom_index = int(np.round(bottom, 2) * 100 - 1) if bottom > 0 and bottom < 1 else 0 if bottom == 0 else 99
        sum_prob = 0
        if up_index == bottom_index:
            return sum_prob
        for i in range(bottom_index, up_index + 1):
            sum_prob += cumulate_prob[i]
        sum_prob /= np.sum(cumulate_prob)
        return sum_prob

#计算进入市场1 / 2 的平均action，去掉未进入市场的action
def mean_action(actions : list, trader_type : str, market_id : int):
    sum = 0
    count = 0
    #market_id = 0.3 if market_id == 0 else 0.6 if market_id == 1 else 0.9
    market_id -= 1  # 要与action匹配
    for i in range(len(actions)):
        if actions[i][0] != market_id: #进入的不是指定市场
            continue
        #if (actions[i] < 0 and trader_type == 'B') or (actions[i] > 1 and trader_type == 'S'): # 未进入市场
            #continue
        else:
            sum += actions[i][1]
            count += 1
    if count == 0:
        return 0
    else:
        return sum / count

def max_action(actions : list, market_id : int): # 指定市场中，buyer 最高价
    if len(actions) == 0:
        return 0
    m = - np.Inf
    market_id -= 1  # 要与action匹配
    #market_id = 0.3 if market_id == 0 else 0.6 if market_id == 1 else 0.9
    for i in range(len(actions)):
        if int(actions[i][0]) != market_id: #不是进入指定市场
            continue
        elif m < actions[i][1]:
            m = actions[i][1]
    if m == -np.inf:
        return 0
    else:
        return m

#计算actions中最低价，存在None
def min_action(actions : list, market_id : int): # 指定市场中，seller 最低价
    if len(actions) == 0: return 0
    n = np.Inf
    market_id -= 1  # 要与action匹配
    #market_id = 0.3 if market_id == 0 else 0.6 if market_id == 1 else 0.9
    for i in range(len(actions)):
        if actions[i][0] != market_id: #not enter market
            continue
        elif n > actions[i][1]:
            n = actions[i][1]
    if n == np.inf:
        return 0
    else:
        return n

#获取进入不同市场的卖家和买家的人数
def get_traders_number(actions, market_id):
    number = 0
    #market_id = 0.3 if market_id == 0 else 0.6 if market_id == 1 else 0.9
    market_id -= 1  # 要与action匹配
    for i in range(len(actions)):
        if actions[i][0] == market_id:  # 进入了指定市场
            number += 1
    return number

def get_discrete_action(market_number : int):
    action = []
    delta = 0.01
    if market_number == 1:# 每个动作一维数组就可以表示
        action.append(None)
        for i in range(int(1 / delta) + 1):
            action.append(delta * i)
    if market_number == 2: # 每个动作需要二维数据来表示
        action.append([None, None])
        for i in range(int(1 / delta) + 1):
            action.append([delta * i, None])
        for j in range(int(1 / delta) + 1):
            action.append([None, delta * j])
    return action

def train(env : Env, sellers : List[Agent], buyers : List[Agent], market_number : int):
    sellers_type, buyers_type = get_type(sellers, buyers)
    SELLER_NUMBER = int(Config.get("Agent", "Seller"))
    BUYER_NUMBER = int(Config.get("Agent", "Buyer"))
    for i in range(int(Config.get("Common", "Episodes"))):
        print("\033[31;1m =============Episodes%d==================\033[0m" %i)
        s = env.initState()
        # 对每个agent初始化action
        seller_combination_action = [[0, 0]] * SELLER_NUMBER
        buyer_combination_action = [[0, 0, 0]] * BUYER_NUMBER
        a_t_sellers = [0] * SELLER_NUMBER
        a_t_buyers = [0] * BUYER_NUMBER
        #original action, 未加上noise
        a_origin_sellers = [0] * SELLER_NUMBER
        a_origin_buyers = [0] * BUYER_NUMBER
        #noise_t
        noise_seller = [0] * SELLER_NUMBER
        noise_buyer = [0] * BUYER_NUMBER
        # var
        var_sellers = [0] * SELLER_NUMBER
        var_buyers = [0] * BUYER_NUMBER
        # 初始的state
        state_sellers = [s] * SELLER_NUMBER
        state_buyers = [s] * BUYER_NUMBER
        # 初始化total_reward
        reward_sellers = [0] * SELLER_NUMBER
        reward_buyers = [0] * BUYER_NUMBER
        # 组合离散-连续动作参数
        actor_output_sellers = [[0, 0, 0, 0]] * SELLER_NUMBER
        actor_output_buyers = [[0, 0, 0, 0]] * BUYER_NUMBER
        #
        ep_sellers_rewards = [0] * SELLER_NUMBER
        ep_buyers_rewards = [0] * BUYER_NUMBER
        #是否成交
        isTran_sell = [int] * SELLER_NUMBER
        isTran_buy = [int] * BUYER_NUMBER
        #
        sell_own_avr_pricing = [0] * SELLER_NUMBER
        sell_oppo_avr_pricing = [0] * BUYER_NUMBER
        buy_own_avr_pricing = [0] * SELLER_NUMBER
        buy_oppo_avr_pricing = [0] * BUYER_NUMBER

        count = 0
        # 计算action
        start = time.time()
        for j in range(int(Config.get('Common', 'Max_EP_Steps'))):
            count += 1
            '''
            state: 
            s[0]: 平均成交价格的归一化
            s[1]: 概率加权比
            s[2]: 概率加权比
            s[3]: 对手最佳出价
            s[4]: 同类最佳出价
            s[5]: 同类平均价格
            s[6]: 对手平均价格
            #s[7]: 是否成交  1-成交，0-不成交
            s[8]: 上一轮成交价格
            '''
            if j == 0:
                #init_pricing = np.round(random.uniform(0, 1), 2)
                for k in range(len(sellers)):
                    temp = np.inf if sellers[k].type == 0 else (1 - sellers[k].type) / sellers[k].type
                    seller_state = [1 - env.avr_pricing[0], 1.0 / (1 + np.exp(-temp)),
                                    random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1),
                                    random.uniform(0, 1),
                                    random.uniform(0, 1), env.tran_pricing[0]] # 1 if random.random() < 0.5 else 0,
                    state_sellers[k] = seller_state
                    # todo 参数解释
                    # todo 选择的离散-连续对【a,b】， 所有的action组合， 连续动作的原始param, 加噪音
                    seller_combination_action[k], actor_output_sellers[k], a_origin_sellers[k], noise_seller[k] = sellers[k].chooseAction(seller_state, sellers[k].trader_type, sellers[k].type, i, market_number,env.market1_pricing, env.market2_pricing)#todo 此处每个元素是组合的
                    if seller_combination_action[k][0] == -1: # 表示不进入市场
                        a_t_sellers[k] = None
                    else:
                        market_id = seller_combination_action[k][0]
                        a_t_sellers[k] = seller_combination_action[k][1] # 存放实际意义的action todo 未修改
                    #index_sellers[k] = index_action #存储DQN的输出值

                    temp1 = np.inf if 1 - buyers[k].type == 0 else buyers[k].type / (1 - buyers[k].type)
                    buyer_state = [env.avr_pricing[0], 1.0 / (1 + np.exp(-(temp1))),  random.uniform(0, 1),
                                    random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1),
                                   random.uniform(0, 1), env.tran_pricing[0]]  # 0 if random.random() < 0.5 else 1,
                    state_buyers[k] = buyer_state
                    buyer_combination_action[k], actor_output_buyers[k], a_origin_buyers[k], noise_buyer[k] = buyers[k].chooseAction(buyer_state, buyers[k].trader_type, buyers[k].type, i, market_number, env.market1_pricing, env.market2_pricing)
                    if buyer_combination_action[k][0] == -1: # no enter market
                        a_t_buyers[k] = None
                    else:
                        market_id = buyer_combination_action[k][0]
                        a_t_buyers[k] = buyer_combination_action[k][1]
            else:
                for k in range(len(sellers)):
                    seller_combination_action[k], actor_output_sellers[k], a_origin_sellers[k], noise_seller[k] = sellers[k].chooseAction(state_sellers[k], sellers[k].trader_type, sellers[k].type, i, market_number, env.market1_pricing, env.market2_pricing)
                    if seller_combination_action[k][0] == -1: #no enter
                        a_t_sellers[k] = None
                    else: # 进入市场
                        market_id = seller_combination_action[k][0]
                        a_t_sellers[k] = seller_combination_action[k][1]

                    buyer_combination_action[k], actor_output_buyers[k], a_origin_buyers[k], noise_buyer[k] = buyers[k].chooseAction(state_buyers[k], buyers[k].trader_type, buyers[k].type, i, market_number, env.market1_pricing, env.market2_pricing)
                    if buyer_combination_action[k][0] == -1: # 不进入市场
                        a_t_buyers[k] = None
                    else:  # 进入市场  0 or 1
                        market_id = buyer_combination_action[k][0]
                        a_t_buyers[k] = buyer_combination_action[k][1]

            #此处是测试，将seller的state和original_action 输出到一个文件,测试是否是state的问题
            if j % 20 == 0:
                t1 = open("seller.txt", "a+")
                print("======episode%d=======\n" % i, "states:\n", state_sellers, '\n', "origin_actions:\n", a_origin_sellers,'\n', "action:\n", a_t_sellers, file=t1)

            #计算reward和next_state
            #todo，计算两个市场进行交易的价格和人数
            m1_tran_pricing = 0
            m2_tran_pricing = 0
            reward_sellers, reward_buyers, m1_tran_pricing, m2_tran_pricing = env.all_market_reward(seller_combination_action, buyer_combination_action,
                                                               sellers_type, buyers_type, j)
            if m1_tran_pricing is None and m2_tran_pricing is None:  # todo check, tran_pricing 为None，可能的情况是没有一个trader进入市场或者是没有匹配到的agent,此情况下表示这个市场交易价格为0
                continue
            if m1_tran_pricing is not None:
                env.tran_pricing[0] = m1_tran_pricing
                ep_sellers_rewards = list_add(reward_sellers, ep_sellers_rewards)  # 累积收益
                env.compute_prob_distribution(np.round(env.tran_pricing[0], 2), 1)  # 更新pricing的概率分布
                env.avr_pricing[0] = (env.avr_pricing[0] * count + env.tran_pricing[0]) / (count + 1)  # 更新市场1的平均价格
            if m2_tran_pricing is not None:
                env.tran_pricing[1] = m2_tran_pricing
                ep_buyers_rewards = list_add(reward_buyers, ep_buyers_rewards)
                env.compute_prob_distribution(np.round(env.tran_pricing[1], 2), 2)  # 更新pricing的概率分布
                env.avr_pricing[1] = (env.avr_pricing[1] * count + env.tran_pricing[1]) / (count + 1)  # 更新市场2的平均价格

            market1_sellers_avr_actions = mean_action(seller_combination_action, 'S', 1)  # todo  检查
            market1_buyers_avr_actions = mean_action(buyer_combination_action, 'B', 1)
            market2_sellers_avr_actions = mean_action(seller_combination_action, 'S', 2)
            market2_buyers_avr_actions = mean_action(buyer_combination_action, 'S', 2)

            #get next_state
            market1_bid = max_action(buyer_combination_action, 1)  # todo check
            market1_ask = min_action(seller_combination_action, 1)
            market2_bid = max_action(buyer_combination_action, 2)
            market2_ask = min_action(seller_combination_action, 2)
            seller_number_in_market = [get_traders_number(seller_combination_action, 1),
                                       get_traders_number(seller_combination_action, 2)]
            buyer_number_in_market = [get_traders_number(buyer_combination_action, 1),
                                      get_traders_number(buyer_combination_action, 2)]
            for k in range(len(sellers)):
                if seller_combination_action[k][0] != -1:  # 进入市场
                    discrete_action = int(seller_combination_action[k][0])  # 0.6 or 0.9
                    m_id = discrete_action
                    #m_id = 0 if discrete_action == 0.6 else 1  # 为0 / 1 表示进入市场1 or 2
                    #todo 分市场是否进行交易
                    if (m_id == 0 and m1_tran_pricing is None) or (m_id == 1 and m2_tran_pricing is None):  #此处表示若enter market，但是没有成交，就不更新state
                        continue
                    s_0 = 1 - env.avr_pricing[m_id]
                    up_1 = integral(env.market_prob[m_id], True, env.tran_pricing[m_id], sellers[k].type, 1)
                    bottom_1 = integral(env.market_prob[m_id], False, env.tran_pricing[m_id], 0, sellers[k].type)
                    s_1 = np.inf if bottom_1 == 0 else up_1 / bottom_1
                    s_1 = 1.0 / (1 + np.exp(-s_1))#将其sigmod() 缩放到[0,1]

                    up_2 = integral(env.market_prob[m_id], True, env.tran_pricing[m_id], seller_combination_action[k][1], 1)
                    bottom_2 = integral(env.market_prob[m_id], False, env.tran_pricing[m_id], 0, seller_combination_action[k][1])
                    s_2 = np.inf if bottom_2 == 0 else up_2 / bottom_2
                    s_2 = 1.0 / (1 + np.exp(-s_2))#将其sigmod() 缩放到[0,1]

                    s_3 = 1 - market1_bid if m_id == 0 else 1 - market2_bid
                    s_4 = 1 - market1_ask if m_id == 0 else 1 - market2_ask
                    new_sellers_avr_actions = market1_sellers_avr_actions if m_id == 0 else market2_sellers_avr_actions
                    s_5 = 0
                    if seller_number_in_market[m_id] == 1:  # 市场人数为1,表示没有其他人进入，同类的平均动作为0
                        s_5 = 0
                    else:
                        s_5 = (new_sellers_avr_actions * seller_number_in_market[m_id] - seller_combination_action[k][1]) / (seller_number_in_market[m_id] - 1)  # todo sell number 需要修改
                    s_6 = market1_buyers_avr_actions if m_id == 0 else market2_buyers_avr_actions
                    #s_7 = isTran_sell[k]
                    s_8 = env.tran_pricing[m_id]
                    seller_next_state = [s_0, s_1, s_2,s_3, s_4, s_5, s_6, s_8]
                    sellers[k].remember_for_rl(state_sellers[k], actor_output_sellers[k], reward_sellers[k], seller_next_state)
                    state_sellers[k] = seller_next_state
                # 硬编码。。。todo
                if seller_combination_action[k][0] == -1:  # 不进入市场，0.3 todo check, 未进入市场，如何更新state。
                    id = 0 if random.random() < 0.5 else 1
                    s_0 = 1 - env.avr_pricing[id]
                    up_1 = integral(env.market_prob[id], True, env.tran_pricing[id], sellers[k].type, 1)
                    bottom_1 = integral(env.market_prob[id], False, env.tran_pricing[id], 0, sellers[k].type)
                    s_1 = np.inf if bottom_1 == 0 else up_1 / bottom_1
                    s_1 = 1.0 / (1 + np.exp(-s_1))  # 将其sigmod() 缩放到[0,1]
                    s_2 = state_sellers[k][2] #由于没有进入市场没有报价， 所以s_2不变
                    s_3 = 1 - market1_bid if id == 0 else 1 - market2_bid
                    s_4 = 1 - market1_ask if id == 0 else 1 - market2_ask
                    s_5 = market1_sellers_avr_actions if id == 0 else market2_sellers_avr_actions
                    s_6 = market1_buyers_avr_actions if id == 0 else market2_buyers_avr_actions
                    #s_7 = 0
                    s_8 = env.tran_pricing[id]
                    seller_next_state = [s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_8]
                    sellers[k].remember_for_rl(state_sellers[k], actor_output_sellers[k], reward_sellers[k], seller_next_state)
                    state_sellers[k] = seller_next_state

                if buyer_combination_action[k][0] != -1: # 买家进入市场
                    discrete_action = buyer_combination_action[k][0]  # 0.6 or 0.9
                    m_id = int(discrete_action)
                    #m_id = 0 if discrete_action == 0.6 else 1
                    # todo 分这个市场是否进行了交易
                    b_0 = env.avr_pricing[m_id]
                    divisor_1 = integral(env.market_prob[m_id], False, env.tran_pricing[m_id], 0, buyers[k].type)
                    divisor_2 = integral(env.market_prob[m_id], True, env.tran_pricing[m_id], buyers[k].type, 1)
                    b_1 = np.inf if divisor_2 == 0 else divisor_1 / divisor_2

                    b_1 = 1.0 / (1 + np.exp(-b_1))  # 映射到0-1

                    divisor_3 = integral(env.market_prob[m_id], False, env.tran_pricing[m_id], 0, seller_combination_action[k][1])
                    divisor_4 = integral(env.market_prob[m_id], True, env.tran_pricing[m_id], a_t_buyers[k], 1)
                    b_2 = np.inf if divisor_4 == 0 else divisor_3 / divisor_4
                    b_2 = 1.0 / (1 + np.exp(-b_2))# 映射到0-1
                    b_3 = market1_ask if m_id == 0 else market2_ask
                    b_4 = market1_bid if m_id == 0 else market2_bid
                    new_buyers_avr_actions = market1_buyers_avr_actions if m_id == 0 else market2_sellers_avr_actions
                    b_5 = 0
                    if buyer_number_in_market[m_id] == 1:  # 没有别的buyer进入，同类action=0
                        b_5 = 0
                    else:
                        b_5 = (new_buyers_avr_actions * buyer_number_in_market[m_id] - a_t_buyers[k]) / (buyer_number_in_market[m_id] - 1)  # todo
                    b_6 = market1_sellers_avr_actions if m_id == 0 else market2_sellers_avr_actions
                    #b_7 = isTran_buy[k]
                    b_8 = env.tran_pricing[m_id]
                    buyer_next_state = [b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_8]
                    buyers[k].remember_for_rl(state_buyers[k], actor_output_buyers[k], reward_buyers[k], buyer_next_state)
                    state_buyers[k] = buyer_next_state

                if a_t_buyers[k] is None:   # 不进入市场,
                    id = 0 if random.random() < 0.5 else 1
                    b_0 = env.avr_pricing[id]
                    divisor_1 = integral(env.market_prob[id], False, env.tran_pricing[id], 0, buyers[k].type)
                    divisor_2 = integral(env.market_prob[id], True, env.tran_pricing[id], buyers[k].type, 1)
                    b_1 = np.inf if divisor_2 == 0 else divisor_1 / divisor_2
                    b_1 = 1.0 / (1 + np.exp(-b_1))  # 映射到0-1
                    b_2 = state_buyers[k][2]
                    b_3 = market1_ask if id == 0 else market2_ask
                    b_4 = market1_bid if id == 0 else market2_bid
                    b_5 = market1_buyers_avr_actions if id == 0 else market2_buyers_avr_actions
                    b_6 = market1_sellers_avr_actions if id == 0 else market2_sellers_avr_actions
                    #b_7 = 0
                    b_8 = env.tran_pricing[id]
                    buyer_next_state = [b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_8]
                    buyers[k].remember_for_rl(state_buyers[k], actor_output_buyers[k], reward_buyers[k], buyer_next_state)
                    state_buyers[k] = buyer_next_state

            if j % 20 == 0:
                for sell in sellers:
                    #sell.update_train(env.market1_pricing, env.market2_pricing)
                    sell.update_strategy()
                    #sell.update_strategy1()
                for buy in buyers:
                    #buy.update_train(env.market1_pricing, env.market2_pricing)
                    buy.update_strategy()
                    #buy.update_strategy1()

                # print 选择market1 market2 no_market 的人数

                #seller and buyer 权重
                f = open('action.txt', 'a+')
                print('=======================step=%d==================\n' % j, 'RewardSeller:', reward_sellers,
                      '\n',
                      'RewardBuyers:', reward_buyers, '\n', 'SellerAction:', a_t_sellers, '\n', 'BuyerAction:',
                      a_t_buyers,
                      '\n', 'SellersOriginalAction:', a_origin_sellers, '\n', 'BuyerOriginalAction:', a_origin_buyers,
                      '\n', "Seller_Noise:", noise_seller, '\n', "Buyer_noise:", noise_buyer, '\n', "VarSellers",
                      var_sellers, '\n', 'VarBuyers', var_buyers, file=f)

            if j == int(Config.get('Common', 'Max_EP_Steps')) - 1:   # 一轮完毕
                print('\033[32;1m =======================enter in, expisode = %d==================\033[0m\n' %i)
                print('\033[32;1m =======================step=%d==================\033[0m\n' % i, 'RewardSeller:', reward_sellers,'\n',
                      'RewardBuyers:', reward_buyers, '\n', 'sell_avr_reward:', avr_reward(ep_sellers_rewards), '\n', 'buyer_avr_reward:', avr_reward(ep_buyers_rewards), '\n',
                      'SellerAction:', seller_combination_action, '\n', 'BuyerAction:', buyer_combination_action,
                      '\n', 'Original_seller:', a_origin_sellers, '\n', 'Original_buyer:', a_origin_buyers, '\n', "Noise_seller:", noise_seller,'\n', 'NOise_buyer:','\n', noise_buyer)

                doc = open('output_loss.txt', 'w')
                for se in range(len(sellers)):
                    print("=====seller%d======\n" % se, sellers[se].loss, file=doc)
                    sellers[se].loss.clear()
                for bu in range(len(buyers)):
                    print("======buyer%d=======\n" % bu, buyers[bu].loss, file=doc)
                    buyers[bu].loss.clear()

                grad = open('grad.txt', 'w')
                for se in range(len(sellers)):
                    print("=====seller%d======\n" % se, sellers[se].grad, file=grad)
                    sellers[se].grad.clear()
                for bu in range(len(buyers)):
                    print("======buyer%d=======\n" % bu, buyers[bu].grad, file=grad)
                    buyers[bu].grad.clear()

                min_strategy = open('strategy_by_grad.txt', 'w')
                for se in range(len(sellers)):
                    print("=====seller%d======\n" % se, sellers[se].test, file=min_strategy)
                    sellers[se].test.clear()
                for bu in range(len(buyers)):
                    print("======buyer%d=======\n" % bu, buyers[bu].test, file=min_strategy)
                    buyers[bu].test.clear()

                end = time.time()
                print("第%d episode用时:" % i, end - start)
        if np.mod(i + 1, 10) == 0:
            print("we save model")
            for k in range(len(sellers)):
                #sellers[k].model_prediction.save_weights("q_prediction_" + str(k) + ".h5", overwrite=True)
                sellers[k].actor.model.save_weights("continuous_actor_" + str(k) + ".h5", overwrite=True)
                sellers[k].critic.model.save_weights("continuous_critic_" + str(k) + ".h5",overwrite=True)
                #buyers[k].model_prediction.save_weights("q_prediction_" + str(k + SELLER_NUMBER) + ".h5", overwrite=True)
                buyers[k].actor.model.save_weights("continuous_actor_" + str(k + SELLER_NUMBER) + ".h5",
                                                          overwrite=True)
                buyers[k].critic.model.save_weights("continuous_critic_" + str(k + SELLER_NUMBER) + ".h5", overwrite=True)

    return

def generate_uniform_distrbution_type(traders_number, trader_type):
    ttype = []
    for i in range(traders_number):
        ttype.append(np.round(random.uniform(0, 1), 2))

    if trader_type == 'S':  #从小到大
        return sorted(ttype)
    else:  # 从大到小
        return sorted(ttype, reverse=True)

def main():
    with tf.Session() as sess:
        env = Env('bal', 2, 0.7, 0.5, 0.01)  #市场定价分别为0, 0.5
        np.random.seed(int(Config.get('Utils', 'Seed')))
        tf.set_random_seed(int(Config.get('Utils', 'Seed')))
        market_number = 2
        step = float(Config.get('Agent', 'step'))
        state_dim = 8
        # 初始化多个agent
        action_dim = [2, 2]
        sellers = []
        buyers = []
        Snumber = int(Config.get('Agent', 'Seller'))
        Bnumber = int(Config.get('Agent', 'Buyer'))
        step = 1 / Snumber
        #stype = [0.18, 0.26, 0.29, 0.41, 0.53, 0.6, 0.7, 0.8, 0.9, 1.0]
        #btype = [0.95, 0.73, 0.63, 0.54, 0.37, 0.47, 0.3, 0.2, 0.1, 0.0]
        stype = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        btype = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0 ]
        f = open("type.txt", 'a+')
        print("seller type:", stype, '\n', "buyer_type:", btype, file=f)
        #Stype = generate_uniform_distrbution_type(Snumber)
        #Btype = generate_uniform_distrbution_type(Bnumber)
        for i in range(Snumber):
            before = time.time()
            agent = Agent(sess, env, stype[i], 'S', i + 1, state_dim, action_dim)
            sellers.append(agent)
            after = time.time()
            print("第"+str(i) + "个卖家", after - before)
        for j in range(Bnumber):
            before = time.time()
            agent = Agent(sess, env, btype[j], 'B', j + Snumber, state_dim, action_dim)
            buyers.append(agent)
            after = time.time()
            print("第" + str(j) + "个买家", after - before)

        sess.run(tf.global_variables_initializer())
        # Now load the weight
        print("Now we load the weight")
        try:
            for i in range(len(sellers)):
                #sellers[i].model_prediction.load_weights("q_prediction_" + str(i) + ".h5")
                sellers[i].actor.model.load_weights("continuous_actor_" + str(i) + ".h5")
                sellers[i].critic.model.load_weights("continuous_critic_" + str(i) + ".h5")

                #buyers[i].model_prediction.load_weights("q_prediction_" + str(i + Bnumber) + ".h5")
                buyers[i].actor.model.load_weights("continuous_actor_" + str(i + Bnumber) + ".h5")
                buyers[i].critic.model.load_weights("continuous_critic_" + str(i + Bnumber) + ".h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

        train(env, sellers, buyers, market_number)


if __name__ == '__main__':
    main()