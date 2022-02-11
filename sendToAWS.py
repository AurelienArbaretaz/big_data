import paramiko
import os
import json

#load conf file
conf = json.load(open('conf_sendToAWS.json'))
file = os.open(conf["file"], os.O_RDWR)
destination = conf["destination"]
username = conf["username"]
adresse = conf["adresse"]
predictFile = conf["predictFile"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
Key_file = paramiko.RSAKey.from_private_key_file(conf["pathFilePEM"])

#send data.csv to the AWS machine
client.connect(adresse, username=username, port=22, pkey=Key_file)
sftp_client = client.open_sftp()
sftp_client.put(file, destination)

#execute script on the AWS machine 
client.exec_command('python3.10 /home/ec2-user/scriptAWS.py')

#get the prediction of the script
sftp_client.get("/home/ec2-user/"+predictFile,"./"+predictFile)
sftp_client.close()
client.close()
