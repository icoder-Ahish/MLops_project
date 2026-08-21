GROUP="ashis-mlop"
LOCATION="centralindia"
WORKSPACE="mlopssecond"

az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

az extension add -n ml --upgrade || true
