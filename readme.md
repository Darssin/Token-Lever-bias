项目启动参考方式，需要从文件夹中安装 transformers 来实现后续的功能实现

```sh
# 可编辑安装，所有逻辑按照里面的来进行
pip install -e ./transformers

# 一个小脚本，将本地文件改动直接上传服务器的某个路径上
bash sync.sh
```