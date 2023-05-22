# 环境
[工业相机SDK](https://mindvision.com.cn/rjxz/list_12.aspx?lcid=138)

必须是python 3.10，安装第三方库：

`pip3 install -r requirements.txt`

如果需要运行`classifier_training.py`，需要额外安装`torch`以及`torchvision`

# Demo
demo.py实现了装甲板的识别、分类、解算

# 部署
## 自启
0. 确保已经安装`screen`，且`RM23_CV_TJU`在桌面文件夹下，否则需改动`RM23_CV_TJU/autostart.sh`
    ```
    sudo apt install screen
    ```
1. 在`~/.config/autostart/`目录下新建文件`RM23_CV_TJU.desktop`
    ```
    mkdir ~/.config/autostart/
    touch ~/.config/autostart/RM23_CV_TJU.desktop
    ```
2. 使用你喜欢的文本编辑器在`RM23_CV_TJU.desktop`中添加如下内容：
    ```
    [Desktop Entry]
    Type=Application
    Exec=/home/rm/Desktop/RM23_CV_TJU/autostart.sh
    Name=RM23_CV_TJU
    ```
    注意这里假设用户名为`rm`，且`Exec`必须为绝对路径。

    [.desktop文件参数详解](https://specifications.freedesktop.org/desktop-entry-spec/desktop-entry-spec-latest.html)

3. 确保`~/Desktop/RM23_CV_TJU/autostart.sh`文件有执行权限：
    ```
    chmod +x ~/Desktop/RM23_CV_TJU/autostart.sh
    ```
