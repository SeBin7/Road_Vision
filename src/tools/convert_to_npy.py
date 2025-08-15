# convert_calib_to_npy.py
import os
import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse

def convert_calibration_images_to_npy(image_dir, output_path, target_size=(224, 224)):
    """
    캘리브레이션 이미지 폴더를 .npy 파일로 변환
    
    Args:
        image_dir: 캘리브레이션 이미지가 있는 디렉토리 (클래스별 하위폴더 포함)
        output_path: 출력할 .npy 파일 경로
        target_size: 이미지 크기 (width, height)
    """
    print(f"🔄 캘리브레이션 이미지를 .npy로 변환 중...")
    print(f"📂 입력 디렉토리: {image_dir}")
    print(f"📝 출력 파일: {output_path}")
    
    # ImageNet 정규화 (학습 시와 동일하게 적용)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # [0,1] 범위로 변환 및 HWC→CHW
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    all_images = []
    total_count = 0
    
    # 지원하는 이미지 확장자
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # 이미지 파일 수집 (하위 폴더 포함)
    image_paths = []
    for ext in image_extensions:
        # 루트 디렉토리에서 직접 찾기
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        # 하위 디렉토리에서 찾기 (클래스별 폴더 구조 지원)
        image_paths.extend(glob.glob(os.path.join(image_dir, '*', ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, '*', ext.upper())))
    
    image_paths = sorted(list(set(image_paths)))  # 중복 제거 및 정렬
    
    if not image_paths:
        print("❌ 이미지 파일을 찾을 수 없습니다!")
        print(f"   지원 형식: {image_extensions}")
        return None
    
    print(f"🖼️  발견된 이미지 수: {len(image_paths)}")
    
    # 각 이미지 처리
    for i, img_path in enumerate(image_paths):
        try:
            # 이미지 로드
            image = Image.open(img_path).convert('RGB')
            
            # 전처리 적용
            tensor = transform(image)
            
            # numpy 배열로 변환
            numpy_array = tensor.numpy()
            all_images.append(numpy_array)
            total_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"   처리 중... {i + 1}/{len(image_paths)}")
                
        except Exception as e:
            print(f"⚠️  이미지 처리 실패: {img_path} - {str(e)}")
            continue
    
    if not all_images:
        print("❌ 처리된 이미지가 없습니다!")
        return None
    
    # 모든 이미지를 하나의 numpy 배열로 스택 (N, C, H, W)
    calibration_data = np.stack(all_images, axis=0)
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # .npy 파일로 저장
    np.save(output_path, calibration_data)
    
    print(f"✅ 변환 완료!")
    print(f"📊 최종 Shape: {calibration_data.shape}")
    print(f"💾 파일 크기: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"📂 저장 위치: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert calibration images to .npy format for Hailo')
    parser.add_argument('--input-dir', type=str, 
                       default='/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/src_video/calibration_dataset',
                       help='Input directory containing calibration images')
    parser.add_argument('--output-path', type=str, 
                       default='/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/src_video/calibration_data.npy',
                       help='Output .npy file path')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                       help='Target image size [width height] (default: 224 224)')
    
    args = parser.parse_args()
    
    # 입력 디렉토리 존재 확인
    if not os.path.exists(args.input_dir):
        print(f"❌ 입력 디렉토리가 존재하지 않습니다: {args.input_dir}")
        return
    
    # 변환 실행
    result = convert_calibration_images_to_npy(
        args.input_dir, 
        args.output_path, 
        target_size=tuple(args.image_size)
    )
    
    if result:
        print(f"\n🎯 Hailo 양자화 사용법:")
        print(f"hailo optimize model_parsed.har \\")
        print(f"    --output-har-path model_quantized.har \\")
        print(f"    --calib-set-path {result} \\")
        print(f"    --hw-arch hailo8l")

if __name__ == "__main__":
    main()
