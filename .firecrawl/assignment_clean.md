重庆邮电大学本科毕业设计（论文）任务书
题    目
基于深度语义通信（DeepSC）的图像鲁棒传输系统设计与实现
学生姓名
学  号
指导教师
所在单位
通信与信息工程学院
题目类型
□应用型   研究型   □综合型   □其它
是否需要在实验、实习、工程实践和社会调查等社会实践中完成（☑是□否）
一、研究目标
语义通信作为6G的核心技术之一，通过让通信系统直接传输“意义”而非“比特”，在低信噪比和带宽受限环境中具有显著优势。本设计旨在基于深度语义通信框架DeepSC，设计并实现一套面向图像任务的端到端语义编码系统。通过构建语义编码器、可微信道及语义解码器，探索复杂无线信道下图像语义传输的鲁棒性。同时开发可视化软件，实现图像上传、SNR调节、实时重构展示等功能，并对语义通信与传统编码方案进行性能对比与综合分析。
二、主要研究内容和方法
主要研究内容：
基于DeepSC图像语义通信模型，构建卷积与注意力融合的语义编码器，将图像提取成低维语义特征，并通过语义解码器实现重构；研究不同压缩率对重建质量的影响。
在端到端网络中嵌入可微的无线信道模型（AWGN高斯白噪声信道、Rayleigh瑞利衰落信道），使模型能够自适应学习噪声与衰落特性，提高抗干扰能力。
基于Python（Streamlit或PyQt），实现图像上传、SNR调节、语义传输与重构展示、PSNR/SSIM自动计算等功能。
以“JPEG/BPG＋调制信道”方案为基线，对比两者在不同SNR下的重构质量、鲁棒性和延迟，评估语义通信优势。
研究方法：
采用PyTorch搭建端到端语义通信网络，使用Adam优化器与MSE＋SSIM混合损失实现稳定训练。
选用CIFAR-10、Kodak等图像数据集，结合随机增强与多SNR条件的蒙特卡洛仿真，评估模型鲁棒性。
通过模块化设计，将训练好的模型封装为推理接口，并在GUI中实现实时图像展示与性能测试。
通过曲线分析、可视化对比等方式验证模型在不同信道条件下的可靠性与工程可行性。
三、主要考核要求或指标
系统功能要求
系统需提供完整图形用户界面（GUI），支持图像选择、参数设置与结果展示。
支持用户上传本地图像，并自动完成预处理。
提供SNR调节滑块，范围至少覆盖−5 dB～20 dB。
能够实时展示“发送原图”、“DeepSC 重构图”和“传统方案重构图”的对比结果。
界面中自动计算并显示PSNR、SSIM等关键性能指标。
性能量化指标
在极低信噪比（如）的AWGN信道中，DeepSC 系统的PSNR必须优于基线方案至少1.5 dB。
在普通PC上，端到端推理延迟不超过1.5秒。
模型训练的Loss曲线收敛稳定，最终收敛值达到预期水平。
非技术因素分析
隐私与安全性：分析语义特征传输相较比特流在被窃听环境下的解码难度与潜在隐私保护能力。
绿色通信与可持续性：分析语义通信减少重传、冗余编码带来的能耗节省。
四、主要参考文献
H. Xie, Z. Qin, G. Y. Li and B. -H. Juang, "Deep Learning Enabled Semantic Communication Systems," in IEEE Transactions on Signal Processing, vol. 69, pp. 2663-2675, 2021, doi: 10.1109/TSP.2021.3071210.
Qin, Zhijin et al. “Semantic Communications: Principles and Challenges.” ArXiv abs/2201.01389 (2021): n. pag.
E. Bourtsoulatze, D. Burth Kurka and D. Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," in IEEE Transactions on Cognitive Communications and Networking, vol. 5, no. 3, pp. 567-579, Sept. 2019, doi: 10.1109/TCCN.2019.2919300.
Pytorch. PyTorch Documentation and Tutorials \[EB/OL]. https://pytorch.org/docs/stable/index.html.
指导教师签字：
（校外加盖公章）
2022
年
12
月
02
日
专业负责人意见：
□同意立题
□不同意立题
负责人签章：
2022
年
12
月
13
日
学院意见：
□同意立题
□不同意立题
负责人签章：
2022
年
12
月
22
日

