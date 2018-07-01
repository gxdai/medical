#!/bin/bash


#$1: file to transfer
#$2: 0 for from local to server, 1 for from server to local

if [[ $# -eq 1 ]]; then
	echo "Default copy to server"
	scp $1 gxdai@10.10.45.63:/home/gxdai/medical
elif [[ $2 = 1 ]]; then
	echo "copy to local"
	scp gxdai@10.10.45.63:/home/gxdai/medical/$1 ./
fi

