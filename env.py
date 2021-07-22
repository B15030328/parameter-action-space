#env for market's trading
import numpy as np
from PDQN import PDQN as Agent
import configparser
from reason_agent import simpAgent, OriginalAgent
from typing import List
import random

Config = configparser.ConfigParser()
Config.read("./config.ini")
#env.py,主要是为了写对于这个环境来说，环境能提供什么给agent，环境能从agents中获得什么
class Env:
    #env初始，自身属性

    def __init__(self, policy: str, market_number, pricing1, princing2, cost):
        #初始化环境
        self.policy = policy    # discrimination  balanced
        self.market1_pricing = pricing1   #[0,1]
        self.market2_pricing = princing2   #[0,1]
        self.cost = cost         # time cost
        self.count_number_market = [[1] * 100, [1] * 100]
        self.market_prob = [[1] * 100, [1] * 100]  # 2 * 100
        self.avr_pricing = [np.round(random.uniform(0, 1), 2), np.round(random.uniform(0, 1), 2) ]
        self.tran_pricing = [np.round(random.uniform(0, 1), 2), np.round(random.uniform(0, 1), 2)]
        self.market_number = market_number

    '''
    上一轮的动作， 上一轮成交价格， 同类的平均动作，对手的平均动作
    平均成交价格 
    '''
    def initState(self):
        state = np.random.uniform(0, 1, (5, ))
        state[0] = round(state[1], 2)
        state[1] = round(state[1], 2)
        state[2] = round(state[2], 2)
        state[3] = round(state[3], 2)
        state[4] = round(state[4], 2)
        return state


    #获取到上一轮为止所有的成交价格的平均值
    def get_avr_pricing(self):
        return self.avr_pricing

    #计算概率分布函数,todo此处只是更新每个pricing出现的次数
    def compute_prob_distribution(self, pricing, market_id):  # market_id 是进入市场的id
        index = 0
        pricing = 0 if pricing < 0 else 1 if pricing > 1 else pricing
        index = int(pricing * 100 - 1)
        self.market_prob[market_id - 1][index] += 1
        pass

    #计算累积概率分布
    '''
        def compute_cumulate_prob(self):
        self.cumulate_prob[0] = self.prob[0]
        for i in range(1, len(self.prob)):
            self.cumulate_prob[i] = self.cumulate_prob[i - 1] + self.prob[i]
        pass
    '''

    #计算这一轮所有的agent报价之后，均衡匹配，找到均衡匹配的价格区间，交易价格
    #返回的是区间价格范围
    def range_pricing(self, policy, market_pricing, sellers: list, buyers: list):
        sellers = sorted(sellers)
        buyers = sorted(buyers, reverse=True)
        isMatch_sell = []
        isMatch_buy = []
        index = -100 #第一个不能匹配的index
        if policy == "dis":
            pass
        else: # 均衡match
            if len(sellers) == 0 or len(buyers) == 0: # 至少一方没有人
                return -100, -100, -100
            else: #
                length = len(sellers) if len(sellers) < len(buyers) else len(buyers)
                for i in range(length):
                    if sellers[i].action > buyers[i].action:  # unmatch
                        index = i
                        break
                    else: #match
                        sellers[i].isMatch = 1
                        buyers[i].isMatch = 1

                if index == 0: # 一个都没有匹配到
                    return -100, -100, -100
                if index == -100:
                    l = len(sellers) if len(sellers) < len(buyers) else len(buyers)
                    return sellers[l - 1].action, buyers[l - 1].action, l - 1
                else:
                    return sellers[index - 1].action, buyers[index - 1].action, index - 1

    def trading_pricing1(self, policy, market_pricing, sellers_actions : list, buyers_actions : list):
        left, right, index = self.range_pricing(policy, market_pricing, sellers_actions, buyers_actions)
        if policy == "dis":
            pass
        else:
            if left == -100 and right == -100:
                return None, None
            return left + (right - left) * market_pricing, index

    #env返回的是特定agent的reward
    def reward(self, policy: str, market_pricing : int, sellers_actions : list, buyers_actions : list, agent : Agent, action : float):
        trading_pricing, index = self.trading_pricing1(policy, market_pricing, sellers_actions, buyers_actions)
        if agent.trader_type =='S':
            if action < agent.type:
                return self.cost
            else:
                return trading_pricing - agent.type - self.cost
        else:
            if action < agent.type:
                return agent.type - action - self.cost
            else:
                return self.cost


    #返回本次交易所有的agents的payoff, list
    #入参需要agents类型
    #11.13日修改，需要对于new_action增加不同情况的punish
    def total_reward0(self, policy : str, market_pricing : float, sellers_actions, buyers_actions, sellers_type, buyers_types : list):
        sellers = []
        buyers = []
        isTran_sell = [0] * len(sellers_actions)
        isTran_buy = [0] * len(buyers_actions)
        #去掉未进入市场的Agent
        for i in range(len(sellers_actions)):
            if sellers_actions[i] is not None:
                sellers.append(simpAgent(i, sellers_actions[i]))

        for j in range(len(buyers_actions)):
            if buyers_actions[j] is not None: #进入市场
                buyers.append(simpAgent(j, buyers_actions[j]))

        sellers.sort()
        buyers.sort(reverse=True)
        #计算成交价格
        pricing, ind = self.trading_pricing1(policy, market_pricing, sellers, buyers)
        punish = -1
        #首先判断是否存在不合理的action
        sellers_reward = [0] * len(sellers_actions)
        buyers_reward = [0] * len(buyers_actions)

        if pricing == None:#如果一对匹配成功的都没有
            #卖家
            for i in range(len(sellers_actions)):
                if sellers_actions[i] is None: #未进入市场
                    sellers_reward[i] = 0
                elif sellers_actions[i] < sellers_type[i]: #[0, type) 不合理
                    sellers_reward[i] = punish
                else:  #[type, 1] 合理
                    sellers_reward[i] = - self.cost

            #买家
            for j in range(len(buyers_actions)):
                if buyers_actions[j] is None: #未进入市场
                    buyers_reward[j] = 0
                elif buyers_actions[j] <= buyers_types[j]: #[0, type] 出价合理
                    buyers_reward[j] = - self.cost
                else: # （type,1] 不合理
                    buyers_reward[j] = punish

        else: #存在匹配成功的交易者
            #卖家
            for i in range(len(sellers_actions)):
                if sellers_actions[i] is None:  # 未进入市场
                    sellers_reward[i] = 0
                    isTran_sell[i] = 0
                elif sellers_actions[i] < sellers_type[i]: # [0, type)进入市场，但是出价超过范围
                    sellers_reward[i] = punish
                    isTran_sell[i] = 0
                else: # [type, 1]进入市场，出价在范围内
                    if sellers_actions[i] <= sellers[ind].action: #并且匹配成功
                        sellers_reward[i] = pricing - sellers_type[i] - self.cost
                        isTran_sell[i] = 1
                    else: #匹配不成功
                        sellers_reward[i] = -self.cost
                        isTran_sell[i] = 0

            #买家
            for j in range(len(buyers_actions)):
                if buyers_actions[j] is None:  #未进入市场
                    buyers_reward[j] = 0
                    isTran_buy[j] = 0
                elif buyers_actions[j] <= buyers_types[j]: # [0, type]进入市场,出价合理
                    if buyers_actions[j] >= buyers[ind].action:  # 并且匹配成功
                        buyers_reward[j] = buyers_types[j] - pricing - self.cost
                        isTran_buy[j] = 1
                    else:# 未匹配成功
                        buyers_reward[j] = - self.cost
                        isTran_buy[j] = 0
                else: #(type,1] 进入市场，出价超过范围
                    buyers_reward[j] = punish
                    isTran_buy[j] = 0

        return sellers_reward, buyers_reward, isTran_sell, isTran_buy, pricing



    '''
    对于多个市场下，计算每个agent的动作以及收益。
    1.首先需要分出来，是不进入，还是进入1，进入2
    2. 对每个agent记录：id，action, reward
    '''
    def separate_market(self, combination_action_seller, combination_action_buyer, sellers_type, buyers_type):
        no_enter_seller = []
        no_enter_buyer = []
        market_1_seller = []
        market_1_buyer = []
        market_2_seller = []
        market_2_buyer = []

        for i in range(len(combination_action_seller)): # 此处组合action:[discrete, continuous_1, continuous_2]
            if int(combination_action_seller[i][0]) == 0:  # 进入market 1
                market_1_seller.append(simpAgent(i, float(combination_action_seller[i][1]), sellers_type[i], isMatch=0))
                #no_enter_seller.append(simpAgent(i, combination_action_seller[i][1], sellers_type[i]))
            if int(combination_action_seller[i][0]) == 1:  # 进入market 2
                #market_1_seller.append(simpAgent(i, combination_action_seller[i][2], sellers_type[i]))
                market_2_seller.append(simpAgent(i, float(combination_action_seller[i][1]), sellers_type[i], isMatch=0))

        for j in range(len(combination_action_buyer)):
            if int(combination_action_buyer[j][0]) == 0:  #进入市场1
                market_1_buyer.append(simpAgent(j, combination_action_buyer[j][1], buyers_type[j],isMatch=0))
                #no_enter_buyer.append(simpAgent(j, combination_action_buyer[j][1], buyers_type[j]))
            elif int(combination_action_buyer[j][0]) == 1:  # 进入 市场2
                market_2_buyer.append(simpAgent(j, float(combination_action_buyer[j][1]), buyers_type[j], isMatch=0))
                #market_1_buyer.append(simpAgent(j, combination_action_buyer[j][2], buyers_type[j]))
        '''
        for i in range(len(combination_action_seller)):
            if float(combination_action_seller[i][0]) == 0:  # 不进入
                no_enter_seller.append(simpAgent(i, combination_action_seller[i][1], sellers_type[i]))
            elif float(combination_action_seller[i][0]) == 1:  # 进入市场1
                market_1_seller.append(simpAgent(i, combination_action_seller[i][1], sellers_type[i]))
            else:  # enter market 2
                market_2_seller.append(simpAgent(i, combination_action_seller[i][1], sellers_type[i]))

        for j in range(len(combination_action_buyer)):
            if float(combination_action_buyer[j][0]) == 0:
                no_enter_buyer.append(simpAgent(j, combination_action_buyer[j][1], buyers_type[j]))
            elif float(combination_action_buyer[j][0]) == 1:  # 进入 市场1
                market_1_buyer.append(simpAgent(j, combination_action_buyer[j][1], buyers_type[j]))
            else:
                market_2_buyer.append(simpAgent(j, combination_action_buyer[j][1], buyers_type[j]))
        '''
        return no_enter_seller, no_enter_buyer, market_1_seller, market_1_buyer, market_2_seller, market_2_buyer


    #修改reward的大小。
    def take_action(self, a : simpAgent):
        return a.action

    def total_reward(self, policy : str, market_pricing : float, sellers_actions : list, buyers_actions : list, sellers_type, buyers_types : list):
        #sellers = []
        #buyers = []
        #isTran_sell = [0] * len(sellers_actions)
        #isTran_buy = [0] * len(buyers_actions)
        #去掉未进入市场的Agent
        '''
                for i in range(len(sellers_actions)):
            if sellers_actions[i] is not None: #[0, inf] 进入市场
                sellers.append(simpAgent(i, sellers_actions[i]))

        for j in range(len(buyers_actions)):
            if buyers_actions[j] is not None:
                buyers.append(simpAgent(j, buyers_actions[j]))
        '''

        sellers_actions = sorted(sellers_actions)
        buyers_actions = sorted(buyers_actions, reverse=True)
        #计算该市场中的成交价格
        pricing, ind = self.trading_pricing1(policy, market_pricing, sellers_actions, buyers_actions)
        '''
        isMatch_sell = [False] * len(sellers_actions)
        isMatch_buy = [False] * len(buyers_actions)
        for i in range(len(sellers)):
            if sellers[i].isMatch is True:
                index = sellers[i].index  # get id
                isMatch_sell[index] = True

        for j in range(len(buyers)):
            if buyers[j].isMatch == True:
                index = buyers[j].index
                isMatch_buy[index] = True
        punish = -1
        #首先判断是否存在不合理的action
        sellers_reward = [0] * len(sellers_actions)
        buyers_reward = [0] * len(buyers_actions)
'''
        punish = -1
        if pricing is None:#如果一对匹配成功的都没有  todo check reward是否合理
            #卖家
            for i in range(len(sellers_actions)):  # 要价太高说明不想进入市场[0, inf]
                if sellers_actions[i].action is None: #未进入市场
                    sellers_actions[i].reward = 0
                elif sellers_actions[i].action >= sellers_actions[i].type: #[type, 1]合理
                    sellers_actions[i].reward = -self.cost
                else:
                    sellers_actions[i].reward = punish

            #买家
            for j in range(len(buyers_actions)):
                if buyers_actions[j].action is None: #未进入市场 [-inf, 0）
                    buyers_actions[j].reward = 0
                elif buyers_actions[j].action <= buyers_actions[j].type: #[0, type] 出价合理
                    buyers_actions[j].reward = - self.cost
                else: # （type,1] 不合理
                    buyers_actions[j].reward = punish

        else: #存在匹配成功的交易者
            #卖家
            for i in range(len(sellers_actions)):
                if sellers_actions[i].action is None:  # 未进入市场[0, inf]
                    sellers_actions[i].reward = 0
                    sellers_actions[i].isTran = 0
                elif sellers_actions[i].action >= sellers_actions[i].type: #进入market，并且[type, 1]合理
                    if sellers_actions[i].isMatch == 1: #并且匹配成功
                        sellers_actions[i].reward = pricing - sellers_actions[i].type - self.cost
                        sellers_actions[i].isTran = 1
                    else: #匹配不成功
                        sellers_actions[i].reward = - self.cost * np.power((sellers_actions[i].action - sellers_actions[i].type), 2) + (-self.cost)
                        sellers_actions[i].isTran = 0
                else:  # 进入market，但是[0, type）超过范围
                    if sellers_actions[i].isMatch == 1:  # 匹配成功，投机行为需要加大punish
                        sellers_actions[i].reward = punish - sellers_actions[i].type #不仅要给出good，还要punish
                    else: #投机行为，没有匹配成功也要惩罚
                        sellers_actions[i].reward = punish * np.power((sellers_actions[i].type - sellers_actions[i].action), 2) + punish
                    sellers_actions[i].isTran = 0

            #买家
            for j in range(len(buyers_actions)):
                if buyers_actions[j].action is None:  #未进入市场 [-inf, 1]
                    buyers_actions[j].reward = 0
                    buyers_actions[j].isTran = 0
                elif buyers_actions[j].action <= buyers_actions[j].type: # [0, type]进入市场,出价合理
                    if buyers_actions[j].isMatch == 1:  # 并且匹配成功
                        buyers_actions[j].reward = buyers_actions[j].type - pricing - self.cost
                        buyers_actions[j].isTran = 1
                    else:# 未匹配成功
                        buyers_actions[j].reward = - self.cost * np.power((buyers_actions[j].type - buyers_actions[j].action), 2) + (-self.cost)
                        buyers_actions[j].isTran = 0
                else: #(type,1] 进入市场，出价超过范围
                    if buyers_actions[j].isMatch == 1:  # 投机行为
                        buyers_actions[j].reward = punish - buyers_actions[j].type
                    else: #未匹配到也要惩罚
                        buyers_actions[j].reward = punish * np.power((buyers_actions[j].action - buyers_actions[j].type), 2) + punish
                    buyers_actions[j].isTran = 0

        return sellers_actions, buyers_actions, pricing

    def compute(self, count_number : list, prob : list):
        total = np.sum(count_number)
        for i in range(len(count_number)):
            prob[i] = np.round(count_number[i] / sum, 2)
        return prob

    '''
    #计算当agents采取动作之后，下个环境是什么,
    #用来将其存放经验池，并且s -> s_next,
    #此处要对齐state的几个维度,但是如果返回total state是没有意义的，
    #因为每个agent观察到的状态是有差异的，时荣见的代码中，此处入参是
    #所有agent采取的动作，然后计算所有agent的reward,最后返回的是reward
    和所有agent下一个到达的状态。那么对于我这个情况，我需要返回的是所有的买
    卖家的reward和下一个状态
    ：状态有多个维度：包括同类的平均action，对手的平均action，上一轮的成交价格，
    平均成交价格
    count : 代表已经交易过多少轮了
    '''
    def all_market_reward(self, combination_sell_action, combination_buy_action, sellers_type : list, buyers_type : list, count : int):
        no_enter_sellers, no_enter_buyers, market_1_sellers, market_1_buyers, market_2_sellers, market_2_buyers = self.separate_market(combination_sell_action, combination_buy_action, sellers_type, buyers_type)
        market1_s, market1_b, market_pricing1 = self.total_reward(self.policy, self.market1_pricing, market_1_sellers, market_1_buyers, sellers_type, buyers_type)
        market2_s, market2_b, market_pricing2 = self.total_reward(self.policy, self.market2_pricing, market_2_sellers, market_2_buyers, sellers_type, buyers_type)
        sellers_rewards = [0] * len(combination_sell_action)
        buyers_rewards = [0] * len(combination_buy_action)
        for i in range(len(no_enter_sellers)):
            index = no_enter_sellers[i].index
            sellers_rewards[index] = 0
        for i in range(len(market1_s)):
            index = market1_s[i].index
            sellers_rewards[index] = market1_s[i].reward
        for i in range(len(market2_s)):
            index = market2_s[i].index
            sellers_rewards[index] = market2_s[i].reward

        for j in range(len(no_enter_buyers)):
            index = no_enter_buyers[j].index
            buyers_rewards[index] = 0

        for j in range(len(market1_b)):
            index = market1_b[j].index
            buyers_rewards[index] = market1_b[j].reward

        for j in range(len(market2_b)):
            index = market2_b[j].index
            buyers_rewards[index] = market2_b[j].reward
        # 10次 print 选择1 和 2 的人数，类型和action
        if np.mod(count, 10) == 0:
            ff = open('all_traders_action.txt', 'a+')
            m1_s_type = []
            m1_s_action = []
            m1_b_type = []
            m1_b_action = []
            m2_s_type = []
            m2_s_action = []
            m2_b_type = []
            m2_b_action = []
            for i in range(len(market1_s)):
                m1_s_type.append(market1_s[i].type)
                m1_s_action.append(market1_s[i].action)

            for i in range(len(market1_b)):
                m1_b_type.append(market1_b[i].type)
                m1_b_action.append(market1_b[i].action)

            for i in range(len(market2_s)):
                m2_s_type.append(market2_s[i].type)
                m2_s_action.append(market2_s[i].action)

            for i in range(len(market2_b)):
                m2_b_type.append(market2_b[i].type)
                m2_b_action.append(market2_b[i].action)

            print("\033[32;1m=========count:%d===========\033[0m\n\n" % count, "\033[31;1m total_number of choosing market1: %d \033[0m\n" % (len(market1_s) + len(market1_b)),
                  "sellers_type:", m1_s_type, '\n', "sellers_action:", m1_s_action, '\n', "buyers_type:", m1_b_type, '\n', "buyers_action:", m1_b_action,
                  '\n', "\033[31;1m total_number of choosing market2: %d \033[0m\n" % (len(market2_s) + len(market2_b)), "sellers_type:", m2_s_type,'\n',
                  "sellers_action:", m2_s_action, '\n', "buyers_type:", m2_b_type, '\n', "buyers_action:", m2_b_action, file=ff)

        return sellers_rewards, buyers_rewards, market_pricing1, market_pricing2


    #均衡匹配，找到均衡匹配的价格区间，求均衡交易价格
    def matchAlgo(self, matchtype, sell, buy):
        index = 0
        if matchtype == 1:
            for i in range(0, len(sell)):
                if sell[i] <= buy[i]:
                    i = i + 1
                else:
                    index = i
                    break
        #return self.pricing * (buy[index - 1] - sell[index - 1]) + buy[index - 1]
        return sell[index - 1],  buy[index - 1]

    #传入action,type
    def next_state_traders(self, action_sellers, action_buyers, type_sellers, type_buyers):
        owner_avr_action = []
        oppo_avr_action = []
        tran_pricing = self.trading_pricing1(self.policy, self.pricing, action_sellers, action_buyers)
        isTran = []
        next_state_sellers = [[0,0,0,True]] * len(action_sellers)
        next_state_buyers = [[0,0,0,True]] * len(action_buyers)
        for i in range(len(action_sellers)):  #遍历卖家
            next_state_sellers[i][0] = (np.sum(action_sellers) - action_sellers[i]) / (len(action_sellers) - 1)
            next_state_sellers[i][1] = np.mean(action_buyers)
            next_state_sellers[i][2] = tran_pricing
            next_state_sellers[i][3] = type_sellers[i] < tran_pricing

        for i in range(len(action_buyers)):
            next_state_buyers[i][0] = (np.sum(action_buyers) - action_buyers[i]) / (len(action_buyers) - 1)
            next_state_buyers[i][1] = np.mean(action_sellers)
            next_state_buyers[i][2] = tran_pricing
            next_state_buyers[i][3] = type_buyers[i] > tran_pricing

        return next_state_sellers, next_state_buyers
