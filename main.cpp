#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <array>
#include "src/include/model.h"

using namespace std;

int main() {
    // 创建模型实体类
    model m;
    // 调用类中的网络测试方法，查看网络的结构
//    m.net_test();
    // 进行模型训练
    m.train_model();
    // 测试模型准确率
//    m.test_model();
    return 0;
}
