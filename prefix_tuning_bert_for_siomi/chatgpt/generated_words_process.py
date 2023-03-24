import re

f = open('generated_100_words.txt', 'r')
lines = f.readlines()
f.close()

bads = []
for line in lines:
    line = line.strip()
    line = re.sub('^No\.\d*', '' ,line)
    if line not in ['']:
        bads.append(line)

dd = ','.join(bads)

f = open('generated_100_words_better2.txt', 'r')
lines = f.readlines()
f.close()
bad_goods = []
for line in lines:
    line = line.strip()
    bad, good = line.split('->')
    bad = bad.strip()
    good = good.strip()
    bad_goods.append((bad, good))


