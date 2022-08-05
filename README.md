# Real-CUGAN server
> [鼠鼠的模型](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)的 http API

默认使用`CPU`进行计算，如果服务器有`GPU`请自行修改代码。

## 效果

<table>
	<tr>
		<td align="center"><img src="input_dir1/in.png"></td>
		<td align="center"><img src="opt-dir-all-test/in_2x_tile0.png"></td>
	</tr>
	<tr>
		<td align="center"><img src="input_dir1/in_3d.jpg"></td>
		<td align="center"><img src="opt-dir-all-test/out_3d.jpg"></td>
	</tr>
    <tr>
		<td align="center">输入</td>
		<td align="center">输出</td>
	</tr>
</table>

## 命令行参数
使用fastapi

- asgiapp为```server:app```，可以使用各种asgi兼容服务器进行部署，简单的比如```hypercorn server:app --bind 0.0.0.0:80```，也可以使用gunicorn，[参考](https://github.com/synodriver/Docker-Real-CUGAN-Server)
- 会缓存图片到`./tmp`，下次请求相同图片则不进行计算直接返回。
- 服务器内存`8GB`时，最大约可处理`1080p`图片。

## API
#### 必要参数
> GET http://host:port/scale?url=图片链接

> POST http://host:port/scale BODY: 图片
#### 可选参数
> model=[conservative, no-denoise, denoise1x, denoise2x, denoise3x]

> scale=[2, 3, 4]

> tile=[0, 1, 2, 3, 4]

特别地，`scale=[3, 4]`时没有模型`[denoise1x, denoise2x]`

#### 返回
`webp`格式的输出图片
