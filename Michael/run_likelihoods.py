import subprocess
import multiprocessing


def call(cmd):
    subprocess.call(cmd,shell=True)


with open("pulsars.dat",'r') as FILE:
    pulsars = map(lambda x: x.strip(),FILE.readlines())
#pulsars = np.loadtxt("pulsars.dat",dtype=np.str)



def loop_func(i):
    call("python mcmc_likelihood.py %s"%pulsars[i])

    


    

pool = multiprocessing.Pool(6)#None)

RANGE = range(len(pulsars))
#print pulsars[:39]
#raise SystemExit
results = pool.imap(loop_func,RANGE)#,10)#[::-1)
for i,result in enumerate(results):
    continue
    #a = result #null thing
    #a.flush()
pool.close()
pool.join()
