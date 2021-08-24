capital = 2000
trade = 40
for x in range(trade):
    if x % 10 == 0:
        money = 0
    else: 
        money = capital * .1 * 1.2

    capital = (capital - capital*.05) + money

print(capital)