# MEDCHost20_v3.0

## 软件使用方法

- **在此页面上方的releases中有最新版的可执行程序。**
- **v3.0新增功能：修改小球到指定区域判定标准**，v2.x判定标准：球心距离目标点20以内正常计时，20-50之间停止计时；大于50本次计时作废；v3.x判定标准：球心距离目标点10以内正常计时，10-20之间停止计时；大于20本次计时作废；
- **v3.1新增功能：更新小球坐标变换算法**，避免小球坐标错误判定。
- 在第二个弹出框中输入二维码编号，即自己的组号，**此时上位机只识别给定编号的二维码**，以避免不同组别二维码相互干扰。建议第一次调试时先输入-1，即**上位机识别所有编号的二维码**，在上位机主界面**点击“打开相机”之后**，Group文本框中显示当前二维码编号。如果**二维码编号与组号不同**，可以联系学长更换，或者使用默认值-1。
- 主界面出现后，可以按照“v1.0使用指南.md”进行调试，相关操作不变。建议**先依照“v1.0使用指南.md”调试**，调试基本完成后，再用下列功能计时。
- 打开相机后，查看fps显示值，推荐使用组内性能最高的电脑，确保fps大部分时间大于22。
- 进入初赛任务计时功能（初赛任务及计分方式详见“2020精仪系新生赛评分细则.pdf”）：
  - **确保相机和串口正确打开**。
  - **调节ui显示状态**，在"ui显示状态"下拉框中，默认显示模式为**实时**。如果调整为**间断**模式后，fps明显提高，可以使用**间断**模式调试。如果差别很小，可以继续使用**实时**模式调试，观感更好。
  - 在**task下拉框**中选择想要计时的任务。
  - **点击“选择任务”按钮**，下拉框右侧文本框显示即将计时的任务编号，Task0仍为普通调试功能。任务一在点击选择任务按钮后立即开始计时。任务二，在小球距离（100，100）不超过20时，total time显示'ready...'，当小球距离（100，100）超过20时，total time从0开始计时。total time文本框显示任务总用时（与实际用时存在一秒以内的误差，最后统计结果采用精确值），target point显示**目标点坐标**。
  - 当小球到达目标点区域以内，上位机开始记录**小球进入区域的时间**，在current time文本框中显示。
  - 当小球在该区域内时间达到三秒，右下角多行文本框中显示该点已完成，target point显示下一个目标点。如果该任务所有点均完成，右下角多行文本框中显示该任务已完成，同时显示任务总用时。
  - 如果想要中途退出任务，**点击“终止任务”按钮**即可，任务正常完成时不需要点击“终止任务”按钮。
- **上位机不会向Labview发送目标点坐标**，选手可以选择根据上位机信息手动判断，或者在labview中添加判断程序。
