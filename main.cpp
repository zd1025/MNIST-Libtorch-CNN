#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <array>
#include "src/include/model.h"

using namespace std;

int main() {
    // ����ģ��ʵ����
    model m;
    // �������е�������Է������鿴����Ľṹ
//    m.net_test();
    // ����ģ��ѵ��
    m.train_model();
    // ����ģ��׼ȷ��
//    m.test_model();
    return 0;
}
