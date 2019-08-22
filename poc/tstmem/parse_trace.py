from collections import defaultdict
import traceback
import numpy as np
import sys

file = open(sys.argv[1])

ops_names = ( "unk", "ld", "st", "atom", "bar", "membar.cta", "membar.gl", "membar.sys", "converge", "other")
ops_count = [0 for name in ops_names]
tot_count = 0
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
        line = line.replace("hooklib.so unloading.", "")
        fields = line.split(',')
        if len(fields) < 6:
            continue
        wid = int(fields[1], 16)
        bid = wid >> 16
        active = int(fields[2], 16)
        op = int(fields[3], 10)
        static = int(fields[4], 10) == 1
        locid = int(fields[5], 10)
        pos = 0
        fld = 6
        while active != 0:
            if (active & 1) != 0:
                addr = int(fields[fld], 16);
                #print "Addr %lx static:%i bid=%lx"%(addr,static, (bid<<44))
                addr += static * (bid << 44)
                tid = (wid << 5) + pos
                ops_count[op] += 1
                tot_count += 1
                tids[tid] += 1
                wids[wid] += 1
                oid = OP_MAP[op]
                if oid != NONE:
                    if addr < smallest:
                        smallest = addr
                    addrs[(TID_ADDR,oid)][addr][tid] += 1
                    addrs[(WARP_ADDR,oid)][addr][wid] += 1
                    addrs[(BLOCK_ADDR,oid)][addr][bid] += 1
                fld += 1
            active = active >> 1
            pos += 1
    except:
        print "Error parsing: " + line
        print sys.exc_info()[0]
        traceback.print_exc()
        #sys.exit(0)

cnt_addrs = {}
for id in ID_IDS:
    for op in OP_IDS:
        cnt_addrs[(id,op)] = {}
        for addr in addrs[(id,op)].keys():
            cnt_addrs[(id,op)][addr] = len(addrs[(id,op)][addr])

print "\"type\",\"count\""
for i in range(len(ops_names)):
    print "\"%s\",%i,%f" % (ops_names[i],ops_count[i],(0.0+ops_count[i])/(0.0+tot_count))
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

print "\"scope\",\"op\",\"#threads\",\"min\",\"mean\",\"median\",\"max\""
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
            print "\"%s\",\"%s\",%lx,%i" %(ID_NAMES[id],OP_NAMES[op],addr-smallest,len)
print
