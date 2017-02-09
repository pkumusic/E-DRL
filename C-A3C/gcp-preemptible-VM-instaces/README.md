# Content
This directory contains sample script to utilize GCP preemptible VM instances.

1. Copy following two files to home directory.
 * startup-script
 * shutdown-script

2. Set up meta data of GCP (Google Cloud Platform) as follows (for example):
 * key:   startup-script
 * value: following two lines
```sh
#!/bin/bash
sudo -u itsukara /home/itsukara/startup-script >> /var/log/startup.log 
```

 * key:   shutdown-script
 * value: following two lines
```sh
#!/bin/bash
sudo -u itsukara  /home/itsukara/shutdown-script >> /var/log/shutdown.log
```

3. Set up Google Cloud SDK in non-preemptible VM (non-GCP VM is OK)
 * See [Google Cloud SDK Quickstarts for Linux](https://cloud.google.com/sdk/docs/quickstart-linux)

4. Run gcp-restart in non-preemptible VM
```sh
nohup gcp-restart &> log.gcp-restart &
```

# How to start new training
 * copy source files and script files to new directory e.g. pscOHL12041610
 * make tgz file of the directory e.g. pscOHL12041610.tgz
 * change run-option-gym if necessary
 * copy tgz file (e.g. pscOHL12041610.tgz) to the home directory of preemptible VMs (you had better to make some script to copy file to all VMs)
 * change run_dir in startup-script
 * change runset in startup-script if necessary
 * change init-script to extract source directory from tgz file
 * copy startup-script and init-script to the home directory of preemptible VMs
 * shutdown all preemptible VMs (you can shutdown all VMs in GCP console)
 * start all preemptible VMs (you can start all VMs in GCP console)

# Additional scripts
1. Run gcp-check in non-preemptible VM to check status of VMs
```sh
gcp-check
```

2. Run gen-hosts in non-preemptible VM to generate informatin for /etc/hosts
```sh
gen-hosts
```

3. Run gcp-copy in non-preemptible VM to copy output preemptible VM to /var/www/html and generate 00index.html
```sh
nohup gcp-copy &> log.gcp-copy &
```
