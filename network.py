import torch
from torch import nn

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.residual = nn.Sequential(
            BasicConv(channels, channels//2, 1, stride = 1, padding = 0),
            BasicConv(channels//2, channels, 3, stride = 1, padding = 1)
        )
    
    def forward(self, x):
        x_residual = self.residual(x)
        return x + x_residual
    
class Top_Down(nn.Module):
    """
    FPN의 Top_Down Layer
    Lateral Connection과 Upsampling이 Concat한 뒤에 수행
    Lateral Connection : Residual Block이나 Skip Connection 같이 동일 계층이나 다른 계층의 Layer와 연결되어 특징 맵을 더하거나 concat하는 것
    feature map 크기와 개수가 유지된다
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            BasicConv(in_channels, out_channels, 1, stride = 1, padding = 0),
            BasicConv(out_channels, out_channels * 2, 3, stride = 1, padding = 1),
            BasicConv(out_channels * 2, out_channels, 1, stride = 1, padding = 0),
            BasicConv(out_channels, out_channels * 2, 3, stride = 1, padding = 1),
            BasicConv(out_channels * 2, out_channels, 1, stride = 1, padding = 0)
        )

    def forward(self, x):
        return self.conv(x)
    
class YOLOLayer(nn.Module):
    """
    13x13, 26x26, 52x52 feature map에서 예측 수행
    """
    def __init__(self, channels, anchors, num_classes = 20, img_size = 416):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = 0

        self.conv = nn.Sequential(
            BasicConv(channels, channels * 2, 3, stride = 1, padding = 1),
            nn.Conv2d(channels * 2, self.num_anchors * (self.num_classes + 5), 1, stride = 1, padding = 0)
        )
    
    def forward(self, x):
        x = self.conv(x)

        # grid_size = 13 or 26 or 52
        batch_size = x.size(0)
        grid_size = x.size(2)
        device = x.device

        # (batch, 3, 25(coordinate + objectness + num_classes), grid_size, grid_size)
        prediction = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
        # (batch, 3, grid_size, grid_size, 25(coordinate + objectness + num_classes))
        prediction = prediction.permute(0, 1, 3, 4, 2)
        # 메모리 저장 주소 정렬
        prediction = prediction.contiguous()

        # Confidence Score [1 if object else 0]
        obj_score = torch.sigmoid(prediction[...,4])
        # Predicted Class
        pred_cls = torch.sigmoid(prediction[...,5:])

        # transform_outputs 함수에서 사용할 anchor 박스 전처리
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, device = device)

        # Calculate Bounding Box Coordinates
        pred_boxes = self.transform_outputs(prediction)

        # (batch, num_anchors x grid_size x grid_size, 25(coordinate + objectness + num_classes))
        output = torch.cat((pred_boxes.view(batch_size, -1, 4), # coordinate
                            obj_score.view(batch_size, -1, 1), # objectness
                            pred_cls.view(batch_size, -1, self.num_classes)), -1) # num_classes
        
        return output

    def compute_grid_offsets(self, grid_size, device = 'cpu'):
        """
        transform_outputs 함수에서 사용할 anchor 박스 전처리
        """
        self.grid_size = grid_size
        # if grid_size == 13: stride = 32
        # if grid_size == 26: stride = 16
        # if grid_size == 52: stride = 8
        self.stride = self.img_size / self.grid_size

        # (1, 1, grid_size, grid_size)
        # [[[[1, 2, 3, ..., grid_size] x grid_size]]]
        self.grid_x = torch.arange(grid_size, device = device).repeat(1, 1, grid_size, 1).type(torch.float32)
        # [[[[1, 1, ...], [2, 2, ...]... [grid_size, grid_size, ...]]]]
        self.grid_y = torch.arange(grid_size, device = device).repeat(1, 1, grid_size, 1).transpose(3, 2).type(torch.float32)

        # anchor 사이즈를 feature map 크기로 정규화 (0 ~ 1)
        self.scaled_anchors = torch.tensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors], device = device)

        # transform_outputs 함수에서 바운딩 박스의 h, w를 예측할 때 사용
        self.anchor_w = self.scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))

    def transform_outputs(self, prediction):
        """
        Bounding Box 좌표 계산
        prediction = (batch, num_anchors, grid_size, grid_size, 25(coordinate + objectness + num_classes))
        """

        device = prediction.device
        # x, y : coordinates, w : Bounding Box Width, h : Bounding Box Height
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        w = prediction[...,2]
        h = prediction[...,3]
        
        # Grid별 bx, by, bw, bh
        pred_boxes = torch.zeros_like(prediction[...,:4]).to(device)
        pred_boxes[...,0] = x + self.grid_x
        pred_boxes[...,1] = y + self.grid_y
        pred_boxes[...,2] = torch.exp(w) * self.anchor_w
        pred_boxes[...,3] = torch.exp(h) * self.anchor_h

        return pred_boxes * self.stride
    
class YOLOv3(nn.Module):
    def __init__(self, anchors, num_blocks = [1, 2, 8, 8, 4], num_classes = 20):
        super().__init__()

        ## Backbone(Darknet) ##
        self.conv1 = BasicConv(3, 32, 3, stride = 1, padding = 1)
        self.res_block_1 = self._make_residual_block(32, 64, num_blocks[0])
        self.res_block_2 = self._make_residual_block(64, 128, num_blocks[1])
        # 52x52 FPN Lateral Connection
        self.res_block_3 = self._make_residual_block(128, 256, num_blocks[2])
        # 26x26 FPN Lateral Connection
        self.res_block_4 = self._make_residual_block(256, 512, num_blocks[3])
        # 13x13 FPN
        self.res_block_5 = self._make_residual_block(512, 1024, num_blocks[4])
        
        ## Neck ##
        # FPN Top Down
        self.topdown_1 = Top_Down(1024, 512)
        self.topdown_2 = Top_Down(768, 256)
        self.topdown_3 = Top_Down(384, 128)

        # Upsampling
        self.upsample_1 = self._upsampling(512)
        self.upsample_2 = self._upsampling(256)

        ## Head (Dense Prediction) ##
        # 13x13, 26x26, 52x52 Feature Map에서 예측 수행
        self.yolo_1 = YOLOLayer(512, anchors = anchors[2])
        self.yolo_2 = YOLOLayer(256, anchors = anchors[1])
        self.yolo_3 = YOLOLayer(128, anchors = anchors[0])
        
    def forward(self, x):
        # Darknet
        x = self.conv1(x)
        r1 = self.res_block_1(x)
        r2 = self.res_block_2(r1)
        r3 = self.res_block_3(r2)
        r4 = self.res_block_4(r3)
        r5 = self.res_block_5(r4)

        # Neck
        n1 = self.topdown_1(r5)
        n2 = self.topdown_2(torch.cat((self.upsample_1(n1), r4), dim = 1))
        n3 = self.topdown_3(torch.cat((self.upsample_2(n2), r3), dim = 1))

        # Head
        feat_13x13 = self.yolo_1(n1)
        feat_26x26 = self.yolo_2(n2)
        feat_52x52 = self.yolo_3(n3)

        return torch.cat((feat_13x13, feat_26x26, feat_52x52), 1), [feat_13x13, feat_26x26, feat_52x52]
        

    def _make_residual_block(self, in_channels, out_channels, num_block):
        blocks = []

        # DownSampling
        blocks.append(BasicConv(in_channels, out_channels, 3, stride = 2, padding = 1))

        for i in range(num_block):
            blocks.append(ResidualBlock(out_channels))

        return nn.Sequential(*blocks)
    
    def _upsampling(self, in_channels):
        upsample = []

        upsample.append(BasicConv(in_channels, in_channels // 2, 1, stride = 1, padding = 0))
        upsample.append(nn.Upsample(scale_factor = 2))

        return nn.Sequential(*upsample)
        
if __name__ == '__main__':

    anchors = [[(10,13),(16,30),(33,23)],[(30,61),(62,45),(59,119)],[(116,90),(156,198),(373,326)]]
    x = torch.randn(1, 3, 416, 416)
    with torch.no_grad():
        model = YOLOv3(anchors)
        output_cat, output = model(x)
        print(output_cat.size())
        print(output[0].size(), output[1].size(), output[2].size())