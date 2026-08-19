GROUP="ashishsahu8135-rg"
LOCATION="southindia"
WORKSPACE="mlops-first"

az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

az extension add -n ml --upgrade || true
