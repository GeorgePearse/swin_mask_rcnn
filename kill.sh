nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

# more desperate
sudo fuser -v /dev/nvidia* -k
