import itertools
import numpy as np
import numpy.random
from operator import itemgetter
import matplotlib.pyplot as plt
import os

n_samples = 1000
nominal_values = [1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 75000,
           100000, 150000, 225000, 350000, 500000, 1000000]
nominal_next = [2000, 3000, 5000, 10000, 20000, 30000, 50000, 75000,
           100000, 150000, 225000, 350000, 500000, 1000000, None]
post_tax = [671.5, 1343, 2014.5, 3357.5, 6715, 13430, 20145, 33575, 50362.5,
            67150, 97825, 133187.5, 191330, 254555, 465305]
safe_values = [0, 671.5, 1343, 2014.5, 3357.5, 6715, 13430, 20145, 33575, 50362.5,
            67150, 97825, 133187.5, 191330, 254555]
accuracy = [0.99, 0.99, 0.99, 0.87, 0.87, 0.87, 0.8, 0.8, 0.8, 0.62, 0.62, 0.62, 0.375,
            0.375, 0.375]
lose4 = [0, 0, 0, 0, 0, 6715, 6715, 6715, 6715, 6715,
         6715, 6715, 6715, 6715, 6715]
lose3 = [0, 0, 0, 0, 0, 6715, 6715, 6715, 6715, 6715,
         67150, 67150, 67150, 67150, 67150]

level = {}
for n, p, a, l3, l4, next, s in zip(nominal_values, post_tax, accuracy, lose3, lose4,
                                nominal_next, safe_values):
    level[n]= {"nominal": n, "post_tax": p, "accuracy": a,
                            "lose3": l3, "lose4": l4, "nominal_next": next,
                            "safe": s}


options = {"four_lifelines": [True, False],
           "nominal": nominal_values,
           "phone": [True, False],
           "fifty_fifty": [True, False],
           "ask_audience": [True, False],
           "new_question": [True, False]}


keys, values = zip(*options.items())
states = [v for v in itertools.product(*values)]

#  Get rid of unreachable states s[0] = four_lifelines s[5]=new_question
states = [s for s in states if not (s[0] is False
          and s[5] is True)]
states = states[::-1]  #They are in wrong order for dynamic programming
state_dict = dict.fromkeys(states)
for s in state_dict:
    state_dict[s]={}


def get_state(four_lifelines, nominal, phone, fifty_fifty, ask_audience,
              new_question):
    if nominal == None:
        return {"value": post_tax[-1]}
    return state_dict[(four_lifelines, nominal, phone, fifty_fifty,
                       ask_audience, new_question)]


def sample(acc):
    a = 0.9
    b = b=-4*(acc*a-a)/(4*acc - 1)
    return 0.75 * np.random.beta(a, b) + 0.25


def value_fifty_fifty(p, four_lifelines, nominal, phone, fifty_fifty, ask_audience,
          new_question):
    nominal_next = level[nominal]["nominal_next"]

    if p>=0.5:
        2/3 *get_state(four_lifelines, nominal_next, phone, False, ask_audience,
                  new_question)['value'] +  1/3 *get_value(four_lifelines, nominal, phone, False, ask_audience,
                            new_question, prob=p)
    if p>=0.33:
        return 1/3*get_state(four_lifelines, nominal_next, phone, False, ask_audience,
                  new_question)['value'] + 2/3 * get_value(four_lifelines, nominal, phone, False, ask_audience,
                            new_question, prob=0.5)
    else:
        return get_value(four_lifelines, nominal, phone, False, ask_audience,
                  new_question, prob=0.5)




def get_value(four_lifelines, nominal, phone, fifty_fifty, ask_audience,
              new_question, prob=None, simple=True):


    if prob == None:
        prob = sample(level[nominal]["accuracy"])
    if four_lifelines:
        lose = level[nominal]['lose4']
    else:
        lose = level[nominal]['lose3']
    safe = level[nominal]['safe']
    nominal_next = level[nominal]['nominal_next']
    possible=[("safe", safe)]
    guess = prob*get_state(four_lifelines, nominal_next, phone, fifty_fifty, ask_audience,
              new_question)['value'] + (1-prob)*lose
    possible.append(("guess", guess))
    if phone:
        possible.append(("phone",get_state(four_lifelines, nominal_next, False, fifty_fifty, ask_audience,
                  new_question)['value']))
    if ask_audience:
        if nominal < 10000:
            possible.append(("ask_audience",get_value(four_lifelines, nominal, phone, fifty_fifty, False,
                      new_question, prob=np.max([prob, 0.95]))))
        elif nominal < 50000:
            val = 0
            for correctness in [0.5, 0.7, 0.95]:
                val += 1/3 * get_value(four_lifelines, nominal, phone, fifty_fifty, False,
                          new_question, prob=np.max([prob, correctness]))
            possible.append(("ask_audience",val))
        elif nominal < 100000:
            val = 0
            for correctness in [0.3, 0.5, 0.7]:
                val += 1/3 * get_value(four_lifelines, nominal, phone, fifty_fifty, False,
                          new_question, prob=np.max([prob,correctness]))
            possible.append(("ask_audience", val))
        else:
            val = 0
            for correctness in [0.25, 0.35, 0.5]:
                val += 1/3 * get_value(four_lifelines, nominal, phone, fifty_fifty, False,
                          new_question, prob=np.max([prob,correctness]))
            possible.append(("ask_audience", val))
    if new_question:
        possible.append(("new_question",get_state(four_lifelines, nominal, phone, fifty_fifty, ask_audience,
                          False)['value']))
    if fifty_fifty:
        possible.append(("fifty_fifty",
                        value_fifty_fifty(prob, four_lifelines, nominal,
                                          phone, fifty_fifty,
                                          ask_audience, new_question)))
    if simple:
        return max(possible, key=itemgetter(1))[1]
    else:
        return(possible)

def set_value(four_lifelines, nominal, phone, fifty_fifty, ask_audience,
              new_question):

    value =    np.mean([get_value(four_lifelines, nominal, phone, fifty_fifty,
                           ask_audience, new_question)
                        for i in range(n_samples)])
    state_dict[(four_lifelines, nominal, phone, fifty_fifty,
                ask_audience, new_question)]["value"] = value
    return(value)

for (four_lifelines, nominal, phone, fifty_fifty, ask_audience,
              new_question) in states:
    set_value(four_lifelines, nominal, phone, fifty_fifty, ask_audience,
                  new_question)
print(state_dict)

nominal_dicts = {}
pvalues = np.linspace(0.255, 0.995, 1000)
for n in nominal_values:
    nominal_dicts[n] = {}

for k in state_dict.keys():
        (four_lifelines, nominal, phone, fifty_fifty,
                            ask_audience, new_question) = k
        nominal_dicts[nominal][k] = {}
        alternatives = get_value(four_lifelines, nominal, phone, fifty_fifty,
                            ask_audience, new_question, prob=0.5, simple=False)
        for alt, _ in alternatives:
            nominal_dicts[nominal][k][alt] = []
        for p in pvalues:
            possible = get_value(four_lifelines, nominal, phone, fifty_fifty,
                        ask_audience, new_question, p, simple=False)
            for alt, value in possible:
                nominal_dicts[nominal][k][alt].append(value)

colormap = {"safe": "k", "fifty_fifty": "y", "ask_audience": "b", "guess": "r", "phone": "g", "new_question":"c"}

for n in nominal_values:
    if not os.path.exists("figs/" + str(n)):
        os.makedirs("figs/" + str(n))
    for state  in nominal_dicts[n].keys():
        plt.figure()
        possible = nominal_dicts[n][state]
        (four_lifelines, nominal, phone, fifty_fifty,
                    ask_audience, new_question) = state
        title = str(nominal)
        if four_lifelines:
            title =title + " 4_lifelines"
        else:
            title =title + " 3_lifelines"
        if phone:
            title = title + " phone"
        if fifty_fifty:
            title += " fifty"
        if ask_audience:
            title += " audience"
        if new_question:
            title += " new_question"
        plt.title(title)
        for action, action_values in possible.items():
            plt.plot(pvalues,action_values, color=colormap[action], label=action)
        plt.legend()


        plt.xlabel("Certainty")
        if n <= 100000:
            plt.ylim(0,150000)
        else:
            plt.ylim(0,500000)
        plt.savefig("figs/" + str(n)+"/"+str(title).strip(" ") +".jpg")
        plt.close()




