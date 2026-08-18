GROUP="ashishsahu8135-rg"
LOCATION="southindia"
WORKSPACE="mlops-first"

az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

az extension remove -n ml
az extension add -n ml
