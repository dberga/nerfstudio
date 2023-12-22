echo "run this from a remote computer to make viewer working"
if [ -z $1 ]
then
PORT=7007
else
PORT=$1
fi
echo "removing previous connections"
lsof -ti:$PORT | xargs kill -9
echo "connecting ssh port"
ssh -NfL $PORT:localhost:$PORT Imatge@172.20.120.175
