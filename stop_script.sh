script_name=$1
script_full_path="scripts.$script_name"

echo "Checking if $script_full_path is already running"

# check if the script is running - try to get the pid
pid=`ps -ef | grep $script_full_path | grep -v grep | awk '{print $2}'`

# if pid is not empty, kill it, else say it is not running
if [ -n "$pid" ]; then
    echo "$script_name is already running with pid $pid. Killing it."
    kill $pid
else
    echo "$script_name is not running."
fi