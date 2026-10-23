"""Run from the repository root; Python standard library and R dependencies only."""
import json,os,subprocess,time,urllib.request,urllib.error
from pathlib import Path
port=8765
proc=subprocess.Popen(['Rscript','--vanilla','scripts/run_api.R'],env={**os.environ,'HOST':'127.0.0.1','PORT':str(port)},stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
base=f'http://127.0.0.1:{port}'
def call(route,data=None):
    req=urllib.request.Request(base+route,data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=30) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
try:
    for _ in range(60):
        if proc.poll() is not None:raise RuntimeError(proc.stderr.read().decode())
        try:
            status,health=call('/health')
            if status==200:break
        except urllib.error.URLError:time.sleep(.5)
    else:raise RuntimeError('API did not become ready.')
    assert health['model_id']=='xgb-no-id-989'
    assert call('/schema')[0]==200
    payload=json.loads(Path('examples/request.json').read_text())
    status,result=call('/predict',json.dumps(payload).encode())
    assert status==200,result
    assert result['predictions'][0]['class'] in ['Low','Medium','High']
    assert abs(sum(result['predictions'][0]['probabilities'].values())-1)<1e-5
    assert call('/predict',b'{invalid')[0]==400
    invalid=json.loads(json.dumps(payload));invalid['records'][0]['country']='nicaragua'
    assert call('/predict',json.dumps(invalid).encode())[0]==400
    assert call('/predict',json.dumps({'records':payload['records']*101}).encode())[0]==400
    assert call('/predict',b' '*1000001)[0]==413
    print('HTTP health, schema, prediction, invalid JSON, country, batch and size checks passed.')
finally:
    proc.terminate()
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()

    log=proc.stderr.read().decode()
    if log:print(log)
