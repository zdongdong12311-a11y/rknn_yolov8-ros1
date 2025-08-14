适用于香橙派（包含软件安装和驱动更新请前往https://github.com/rockchip-linux/rknn-toolkit2）
经过测试多线程代码在20.04版本的Ubuntu跑不通，但相同代码在22及以上版本可以跑通
代码集合了v8识别和ros发布，经过优化解决了debug问题但对香橙派cpu占用较高，因此适用于作为ros1与v8结合的保底方案。
注：代码中的多线程操作可以忽略
