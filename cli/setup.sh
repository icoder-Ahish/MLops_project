GROUP="ashish-rg"
LOCATION="centralindia"
WORKSPACE="mlopsfour"

az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

az extension add -n ml --upgrade || true
