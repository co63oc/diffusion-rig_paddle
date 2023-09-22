#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo -e "\nBefore you continue, you must send an email to get username and password at https://faces.dmi.unibas.ch/bfm/bfm2017.html."
read -p "Username:" username
read -p "Password:" password
username=$(urle $username)
password=$(urle $password)

git clone https://github.com/TimoBolkart/BFM_to_FLAME.git

cd BFM_to_FLAME/data/

url1="https://faces.dmi.unibas.ch/bfm/bfm2017/restricted/model2017-1_bfm_nomouth.h5"
url2="https://faces.dmi.unibas.ch/bfm/bfm2017/restricted/model2017-1_face12_nomouth.h5"

echo -e "\nDownloading ..."
wget --user=$username --password=$password $url1 -O 'model2017-1_bfm_nomouth.h5' --no-check-certificate --continue 
wget --user=$username --password=$password $url2 -O 'model2017-1_face12_nomouth.h5' --no-check-certificate --continue 
