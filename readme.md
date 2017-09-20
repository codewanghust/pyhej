
## module: genplate
车牌生成器,参考[szad670401](https://github.com/szad670401/end-to-end-for-chinese-plate-recognition)实现,对代码进行了精简,支持py3.

## 建议
建议作为子模块引入到你的项目:
```
git submodule add https://gitee.com/flystarhe/pyhej.git
```

注意,克隆使用子模块的项目,执行`git submodule *`是必要的,否则子模块不可用.比如:
```
git clone https://gitee.com/flystarhe/slyx_mdisp_guess.git
cd slyx_mdisp_guess
git submodule init
git submodule update

git remote set-url origin https://git.medatc.cc/medatc/mdisp-guess.git
git remote add origin-hej https://gitee.com/flystarhe/slyx_mdisp_guess.git
```

技巧,当需要`pull`子模块时,如果你不想在子目录中手动抓取与合并,那么还有种更容易的方式.运行`git submodule update --remote`,Git将会进入子模块然后抓取并更新.