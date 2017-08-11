
## module: genplate
车牌生成器,参考[szad670401](https://github.com/szad670401/end-to-end-for-chinese-plate-recognition)实现,对代码进行了精简,支持py3.依赖项:

```
pip install pillow opencv numpy
```

运行示例:
```
import sys
sys.path.append('/data2')
from pyhej.genplate import genplate
G = genplate.GenPlate()
G.gen_batch(10, './temp', (272,72))
```
