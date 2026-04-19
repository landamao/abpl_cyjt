import json, tempfile, shutil, asyncio, base64, cv2, time, uuid
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from typing import Awaitable, Any
from astrbot.api.event import filter
from astrbot.api.all import (
    Star, Context, logger, Plain, At, Reply, Image,
    AstrBotConfig, AstrMessageEvent, BaseMessageComponent
)
from astrbot.api.star import StarTools

# ---------- 辅助函数 ----------
async def _发送消息(event: AstrMessageEvent, /, *, 文本: str = None, 引用: bool = False, 艾特: bool = False,
                    base图片: str = None, URL图片: str = None, 本地图片: str = None) -> None:
    """构造消息链并发送"""
    消息链 = []
    if 引用:
        消息链.append(Reply(id=event.message_obj.message_id))
    if 艾特:
        消息链.append(At(qq=event.get_sender_id()))
    if 文本:
        消息链.append(Plain(text=文本))
    if base图片:
        消息链.append(Image.fromBase64(base图片))
    if URL图片:
        消息链.append(Image.fromURL(URL图片))
    if 本地图片:
        消息链.append(Image.fromFileSystem(本地图片))
    await event.send(event.chain_result(消息链))

def _构造消息链(event: AstrMessageEvent, /, *, 文本: str = None, 引用: bool = False, 艾特: bool = False,
                base图片: str = None, URL图片: str = None, 本地图片: str = None) -> list[BaseMessageComponent]:
    """构造消息链并发送"""
    消息链 = []
    if 引用:
        消息链.append(Reply(id=event.message_obj.message_id))
    if 艾特:
        消息链.append(At(qq=event.get_sender_id()))
    if 文本:
        消息链.append(Plain(text=文本))
    if base图片:
        消息链.append(Image.fromBase64(base图片))
    if URL图片:
        消息链.append(Image.fromURL(URL图片))
    if 本地图片:
        消息链.append(Image.fromFileSystem(本地图片))
    return 消息链

def _查找图片(event: AstrMessageEvent) -> Awaitable[str] | None:
    """查找消息链中的图片，返回一张图片的base64字符串"""
    for seg in event.get_messages():
        if isinstance(seg, Reply):
            for seg2 in seg.chain:
                if isinstance(seg2, Image):
                    return seg2.convert_to_base64()
        if isinstance(seg, Image):
            return seg.convert_to_base64()
    return None

# ---------- 透视变换 + 合成核心 ----------
def 透视变换并合成(主图路径: str, 底图路径: str, 模板数据: dict, 输出路径: str):
    """
    将主图透视变形到模板定义的四边形区域，并与底图合成（底图在上层）。
    模板数据需包含: left_top_x/y, right_top_x/y, right_bottom_x/y, left_bottom_x/y,
                  template_width, template_height
    """
    # 1. 读取底图（RGBA）
    底图 = PILImage.open(底图路径).convert("RGBA")
    底图尺寸 = 底图.size  # (width, height)
    模板宽 = 模板数据["template_width"]
    模板高 = 模板数据["template_height"]

    # 如果底图尺寸与模板尺寸不一致，则缩放底图（可选，这里选择缩放底图）
    if 底图尺寸 != (模板宽, 模板高):
        底图 = 底图.resize((模板宽, 模板高), PILImage.Resampling.LANCZOS)

    # 2. 构建目标四边形点集（顺序：左上、右上、右下、左下）
    dst_points = [
        [模板数据["left_top_x"], 模板数据["left_top_y"]],
        [模板数据["right_top_x"], 模板数据["right_top_y"]],
        [模板数据["right_bottom_x"], 模板数据["right_bottom_y"]],
        [模板数据["left_bottom_x"], 模板数据["left_bottom_y"]]
    ]

    # 3. 读取主图
    主图 = cv2.imread(主图路径, cv2.IMREAD_UNCHANGED)
    if 主图 is None:
        raise FileNotFoundError(f"无法读取主图: {主图路径}")
    # 转为 RGBA
    if 主图.shape[2] == 3:
        主图 = cv2.cvtColor(主图, cv2.COLOR_BGR2BGRA)
    h, w = 主图.shape[:2]

    # 4. 透视变换
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points_np = np.float32(dst_points)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points_np)
    变形图 = cv2.warpPerspective(主图, matrix, (模板宽, 模板高), flags=cv2.INTER_LINEAR)

    # 5. 转换为 PIL PILImage
    变形图_pil = PILImage.fromarray(cv2.cvtColor(变形图, cv2.COLOR_BGRA2RGBA))

    # 6. 创建透明底层画布
    底层 = PILImage.new("RGBA", (模板宽, 模板高), (0, 0, 0, 0))
    底层.paste(变形图_pil, (0, 0), 变形图_pil)

    # 7. 上层合成底图
    结果 = PILImage.alpha_composite(底层, 底图)

    # 8. 保存
    结果.save(输出路径, format="PNG")

# ---------- 插件主类 ----------
class 创意截图(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 读取 base64 传输配置
        self.base64传输 = config.get('base64传输', True)

        数据目录 = StarTools.get_data_dir()
        模板根目录 = 数据目录 / "模板目录"
        # 确保模板根目录存在
        模板根目录.mkdir(parents=True, exist_ok=True)

        # ----- 复制预设模板 -----
        script_dir = Path(__file__).resolve().parent  # 插件脚本所在目录
        预设模板目录 = script_dir / "预设模板"  # 预设模板文件夹
        if 预设模板目录.exists() and 预设模板目录.is_dir():
            for 模板文件夹 in 预设模板目录.iterdir():
                if 模板文件夹.is_dir():
                    目标路径 = 模板根目录 / 模板文件夹.name
                    if not 目标路径.exists():
                        shutil.copytree(模板文件夹, 目标路径)
                        logger.info(f"复制预设模板：{模板文件夹.name} -> {目标路径}")
                    else:
                        logger.info(f"模板 {模板文件夹.name} 已存在，跳过复制")
        else:
            logger.info(f"未找到预设模板目录：{预设模板目录}，跳过复制")

        # ----- 加载模板 -----
        self.模板映射 = {}
        self.模板缓存 = {}
        for 模板文件夹 in 模板根目录.iterdir():
            if 模板文件夹.is_dir():
                模板名 = 模板文件夹.name
                底图文件 = 模板文件夹 / "底图.png"
                配置文件 = 模板文件夹 / "模板.json"
                if 底图文件.exists() and 配置文件.exists():
                    with open(配置文件, "r", encoding="utf-8") as f:
                        模板数据 = json.load(f)
                    self.模板映射[模板名] = 模板文件夹
                    self.模板缓存[模板名] = {
                        "底图路径": str(底图文件),
                        "模板数据": 模板数据
                    }
                    logger.info(f"加载模板成功：{模板名}")
                else:
                    logger.warning(f"模板文件夹 {模板文件夹} 缺少底图.png 或 模板.json，已跳过")

        # ----- 读取配置 -----
        self.预回复模式:list = config.get('预回复方式', [])
        self.预回复词模板:str = self.获取值('预回复词', '🎨 正在使用模板「{模板名}」制作，请稍等...')
        self.完成回复模式:list = config.get('完成回复方式', [])
        self.完成回复词模板:str = self.获取值('完成回复词', '✅ 使用模板「{模板名}」生成成功')

        self.output_cache_dir = 数据目录 / "output_cache"
        self.output_cache_dir.mkdir(parents=True, exist_ok=True)
        self._clean_old_cache(hours=1)

    def _clean_old_cache(self, hours: float = 1):
        """清理超过指定小时数的缓存文件"""
        now = time.time()
        cutoff = now - hours * 3600
        for f in self.output_cache_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    logger.info(f"删除过期缓存: {f.name}")
                except Exception as e:
                    logger.warning(f"删除缓存失败 {f}: {e}")

    async def _发送完成回复(self, event: AstrMessageEvent, 模板名: str, 图片base64: str = None, 图片路径: str = None):
        """根据配置发送完成回复（图片，可能带文本/引用/艾特）"""
        # 解析模式
        引用 = "引用" in self.完成回复模式
        艾特 = "艾特" in self.完成回复模式
        提示词 = "提示词" in self.完成回复模式

        # 格式化文本
        文本 = None
        if 提示词:
            文本 = self.完成回复词模板.replace("{模板名}", 模板名)

        # 发送图片（根据提供的参数选择 Base64 或本地路径）
        if 图片base64 is not None:
            await _发送消息(event, base图片=图片base64, 文本=文本, 引用=引用, 艾特=艾特)
        elif 图片路径 is not None:
            await _发送消息(event, 本地图片=图片路径, 文本=文本, 引用=引用, 艾特=艾特)
        else:
            # 没有图片，仅发送文本（一般不会发生）
            await _发送消息(event, 文本=文本, 引用=引用, 艾特=艾特)

    @filter.command("模板截图", alias={"截图模板"}, priority=1314)
    async def 模板截图(self, event: AstrMessageEvent, 模板名:str):
        """使用/模板截图 模板名，并附带一张图片，生成模板截图"""
        模板名 = str(模板名)
        # 获取图片base64
        base64图片 = _查找图片(event)
        if not base64图片:
            await _发送消息(event, 文本="❌ 未找到图片，请发送一张图片（或回复图片）并附带模板名称。", 引用=True)
            return

        # 发送预回复（根据配置）
        if not self.预回复模式:
            return

        # 解析模式
        引用 = "引用" in self.预回复模式
        艾特 = "艾特" in self.预回复模式
        # "仅回复"模式：回复=False,艾特=False
        提示词 = "提示词" in self.预回复模式
        # 格式化文本
        if 提示词:
            文本 = self.预回复词模板.replace("{模板名}", 模板名)
            #使用yield更快的将消息发送出去
            yield event.chain_result(_构造消息链(event, 文本=文本, 引用=引用, 艾特=艾特))
        else:
            yield event.chain_result(_构造消息链(event, 引用=引用, 艾特=艾特))

        logger.info(f"【创意截图】开始制作，模板：{模板名}")

        base64图片 = await base64图片
        # 保存临时主图文件
        临时主图路径 = None
        输出路径 = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                临时主图路径 = tmp.name
            # 将base64解码并保存为临时文件
            if "," in base64图片:
                base64图片 = base64图片.split(",")[-1]  # 去除 data:image 前缀
            图片数据 = base64.b64decode(base64图片)
            with open(临时主图路径, "wb") as f:
                f.write(图片数据)

            # 获取模板信息后
            模板信息 = self.模板缓存[模板名]
            底图路径 = 模板信息["底图路径"]
            模板数据 = 模板信息["模板数据"]

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
                输出路径 = tmp2.name

            await asyncio.to_thread(透视变换并合成, 临时主图路径, 底图路径, 模板数据, 输出路径)

            if self.base64传输:
                with open(输出路径, "rb") as f:
                    图片base64_str = base64.b64encode(f.read()).decode('utf-8')
                await self._发送完成回复(event, 模板名, 图片base64=图片base64_str)
                Path(输出路径).unlink()  # base64模式立即删除
            else:
                # 移动到缓存目录
                目标文件名 = f"{uuid.uuid4()}.png"
                目标路径 = self.output_cache_dir / 目标文件名
                shutil.move(输出路径, 目标路径)
                await self._发送完成回复(event, 模板名, 图片路径=str(目标路径))
                # 不删除，下次启动时统一清理

        except Exception as e:
            logger.error(f"处理截图时出错: {e}", exc_info=True)
            await _发送消息(event, 文本=f"❌ 生成失败：{str(e)}", 引用=True, 艾特=True)
        finally:
            # 清理临时主图文件
            if 临时主图路径 and Path(临时主图路径).exists():
                Path(临时主图路径).unlink()
            # 如果输出路径还存在（比如异常时未移动/未删除），则删除
            if 输出路径 and Path(输出路径).exists():
                Path(输出路径).unlink()

    def 获取值(self, key, default=None) -> Any:
        """键不存在或值为空则返回默认值，优先返回配置里的默认值"""
        # 键不存在 → 返回默认值
        if key not in self.config:
            return default

        # 取值并清理字符串
        值 = self.config[key]
        if not isinstance(值, str):
            #不是字符串直接返回，支持空
            return 值

        值 = 值.strip()

        # 值有效 → 直接返回
        if 值:
            return 值

        # 值为空 → 尝试取 schema 默认值
        if key not in self.config.schema:
            return default

        return self.config.schema[key].get('default', default)

    @filter.command("模板预览", alias={"预览模板"})
    async def 模板预览(self, event: AstrMessageEvent, 模板名: str):
        """预览指定模板的底图"""
        模板名 = str(模板名)
        模板信息 = self.模板缓存.get(模板名)
        if not 模板信息:
            await _发送消息(event, 文本=f"❌ 未找到名为「{模板名}」的模板。", 引用=True)
            return
        
        底图路径 = 模板信息["底图路径"]
        模板数据 = 模板信息["模板数据"]
        
        # 构造预览文本信息（可选）
        预览文本 = (
            f"🖼️ 模板「{模板名}」预览\n"
            f"尺寸：{模板数据.get('template_width', '?')} x {模板数据.get('template_height', '?')}\n"
            f"左上：({模板数据.get('left_top_x', '?')}, {模板数据.get('left_top_y', '?')})\n"
            f"右上：({模板数据.get('right_top_x', '?')}, {模板数据.get('right_top_y', '?')})\n"
            f"右下：({模板数据.get('right_bottom_x', '?')}, {模板数据.get('right_bottom_y', '?')})\n"
            f"左下：({模板数据.get('left_bottom_x', '?')}, {模板数据.get('left_bottom_y', '?')})"
        )
        
        # 发送图片（本地路径）及文本信息
        await _发送消息(event, 本地图片=底图路径, 文本=预览文本, 引用=True)
    
    @filter.command("模板列表", alias={"截图模板列表", "模板截图列表"})
    async def 模板列表(self, event: AstrMessageEvent):
        """列出所有可用的模板"""
        模板名列表 = list(self.模板缓存.keys())
        if not 模板名列表:
            await _发送消息(event, 文本="❌ 当前没有任何可用模板。", 引用=True)
            return
        
        回复文本 = "📋 当前可用的模板列表：\n" + "\n".join(f"• {name}" for name in 模板名列表)
        await _发送消息(event, 文本=回复文本, 引用=True)