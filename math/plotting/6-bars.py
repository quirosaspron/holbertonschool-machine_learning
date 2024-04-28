#!/usr/bin/env python3
"""plot a stacked bar graph"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """colums of fruit represent # of fruit per person
    rows of fruit represent the number of apples, bananas,
    oranges, and peaches
    bars width 0.5
    y-axis ticks every 10 units, range 0-80"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    plt.bar([1, 2, 3], fruit[0], color='red', width=0.5,
            label='apples')
    plt.bar([1, 2, 3], fruit[1], color='yellow', width=0.5,
            bottom=fruit[0], label='bananas')
    plt.bar([1, 2, 3], fruit[2], color='#ff8000', width=0.5,
            bottom=fruit[0] + fruit[1], label='oranges')
    plt.bar([1, 2, 3], fruit[3], color='#ffe5b4', width=0.5,
            bottom=fruit[0] + fruit[1] + fruit[2], label='peaches')

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.xticks([1, 2, 3], ['Farrah', 'Fred', 'Felicia'])
    plt.yticks(range(0, 81, 10))
    plt.ylim(0, 80)
    plt.show()
