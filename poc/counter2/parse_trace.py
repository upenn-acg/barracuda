from collections import defaultdict
import numpy as np
import sys

file = open(sys.argv[1])

ops_names = ( "unk", "ld", "st", "atom", "bar", "membar.cta", "membar.gl", "membar.sys", "converge", "other")
ops_count = [0 for name in ops_names]
tot_count = 0
pred_count = 0
tids = defaultdict(int)
wids = defaultdict(int)
(NONE, LOAD, STORE, ATOM) = (0, 1, 2, 3)
OP_IDS = (LOAD, STORE, ATOM)
OP_MAP = (NONE, LOAD, STORE, ATOM, NONE, NONE, NONE, NONE, NONE, NONE)
OP_NAMES = ("none", "load", "store", "atom")
(TID_ADDR, WARP_ADDR, BLOCK_ADDR) = (0, 1, 2)
ID_IDS = (TID_ADDR, WARP_ADDR, BLOCK_ADDR)
ID_NAMES = ("thread", "warp", "block")
addrs = {}
smallest = 0xffffffffffffffff
for id in ID_IDS:
    for op in OP_IDS:
        addrs[(id,op)] = defaultdict(lambda: defaultdict(int))

for line in file:
    if not line.startswith("TRACE"):
        continue
    try:
        line = line.rstrip('\n')
        fields = line.split(',')
        if len(fields) < 6:
            continue
        first = int(fields[1])
        wid = int(fields[2], 16)
        bid = wid >> 16
        tid = (wid << 5) | int(fields[3])
        active = int(fields[4], 16)
        op = int(fields[5], 10)
        pred = int(fields[6], 10) > 0
        addr = int(fields[7], 16)
    
        ops_count[op] += 1
        tot_count += 1
        pred_count += pred > 0
        tids[tid] += 1
        wids[wid] += 1

        oid = OP_MAP[op]
        if oid != NONE:
            if addr < smallest:
                smallest = addr
            addrs[(TID_ADDR,oid)][addr][tid] += 1
            addrs[(WARP_ADDR,oid)][addr][wid] += 1
            addrs[(BLOCK_ADDR,oid)][addr][bid] += 1
    except:
        pass
#        print "Error parsing: " + line
#        print sys.exc_info()[0]
#        sys.exit(0)

cnt_addrs = {}
for id in ID_IDS:
    for op in OP_IDS:
        cnt_addrs[(id,op)] = {}
        for addr in addrs[(id,op)].keys():
            cnt_addrs[(id,op)][addr] = len(addrs[(id,op)][addr])

print "\"type\",\"count\""
for i in range(len(ops_names)):
    print "\"%s\",%i,%f" % (ops_names[i],ops_count[i],(0.0+ops_count[i])/(0.0+tot_count))
print "\"predicated\",%i,%f" % (pred_count, (0.0+pred_count)/(0.0+tot_count))
print "\"total\",%i,%f" % (tot_count, 1.0)
print

print "\"#tids\",%i" % len(tids)
print "\"ins-per-tid-min\",%i" % np.min(tids.values())
print "\"ins-per-tid-max\",%i" % np.max(tids.values())
print "\"ins-per-tid-avg\",%i" % np.mean(tids.values())
print "\"ins-per-wid-min\",%i" % np.min(wids.values())
print "\"ins-per-wid-max\",%i" % np.max(wids.values())
print "\"ins-per-wid-avg\",%i" % np.mean(wids.values())
print

print "\"scope\",\"op\",\"#addrs\",\"min\",\"mean\",\"median\",\"max\""
for id in ID_IDS:
    for op in OP_IDS:
        vals = cnt_addrs[(id,op)].values()
        if len(vals) > 0 :
            print "\"%s\",\"%s\",%i,%f,%f,%f,%f" %(ID_NAMES[id],OP_NAMES[op],len(vals),np.min(vals),np.mean(vals), np.median(vals), np.max(vals))
print

print "\"op\",\"scope\",\"addr\",\"count\""
for id in ID_IDS:
    for op in OP_IDS:
        db = cnt_addrs[(id,op)]
        for addr in sorted(db.keys()):
            len = db[addr]
            print "\"%s\",\"%s\",%li,%i" %(ID_NAMES[id],OP_NAMES[op],addr-smallest,len)
print
