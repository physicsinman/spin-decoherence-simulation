#!/usr/bin/env python3
"""
이전 시뮬레이션 결과 파일 정리 스크립트

사용법:
    python3 cleanup_old_results.py --dry-run    # 삭제할 파일만 확인
    python3 cleanup_old_results.py --execute    # 실제 삭제
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

def get_file_info(filepath):
    """파일 정보 가져오기"""
    stat = filepath.stat()
    return {
        'name': filepath.name,
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime)
    }

def analyze_results_files(directory='results_comparison'):
    """결과 파일 분석"""
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"⚠️  디렉토리가 없습니다: {directory}")
        return
    
    json_files = list(dir_path.glob('*.json'))
    
    if not json_files:
        print("✓ JSON 파일이 없습니다.")
        return
    
    print(f"\n{'='*70}")
    print(f"결과 파일 분석: {directory}")
    print(f"{'='*70}")
    print(f"총 파일 수: {len(json_files)}")
    
    # 파일 크기 계산
    total_size = sum(f.stat().st_size for f in json_files)
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size_mb / 1024
    
    print(f"총 크기: {total_size_mb:.1f} MB ({total_size_gb:.2f} GB)")
    
    # 파일 분류
    all_results_files = [f for f in json_files if 'all_results' in f.name]
    individual_files = [f for f in json_files if 'all_results' not in f.name]
    
    print(f"\n파일 분류:")
    print(f"  • 통합 결과 파일 (all_results_*.json): {len(all_results_files)}개")
    print(f"  • 개별 결과 파일: {len(individual_files)}개")
    
    # 최신 파일 찾기
    if all_results_files:
        latest_all = max(all_results_files, key=lambda f: f.stat().st_mtime)
        latest_time = datetime.fromtimestamp(latest_all.stat().st_mtime)
        print(f"\n최신 통합 파일:")
        print(f"  • {latest_all.name}")
        print(f"  • 수정 시간: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • 크기: {latest_all.stat().st_size / (1024*1024):.1f} MB")
    
    # 개별 파일 그룹화
    if individual_files:
        print(f"\n개별 파일 (샘플):")
        for f in sorted(individual_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size_mb = f.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  • {f.name} ({size_mb:.1f} MB, {mtime.strftime('%Y-%m-%d')})")
    
    return {
        'all_files': json_files,
        'all_results_files': all_results_files,
        'individual_files': individual_files,
        'total_size_mb': total_size_mb
    }

def cleanup_files(directory='results_comparison', dry_run=True, keep_latest=True):
    """파일 정리"""
    info = analyze_results_files(directory)
    
    if not info:
        return
    
    print(f"\n{'='*70}")
    if dry_run:
        print("DRY RUN 모드: 실제로 삭제하지 않습니다")
    else:
        print("실제 삭제 모드: 파일을 삭제합니다!")
    print(f"{'='*70}")
    
    files_to_delete = []
    files_to_keep = []
    
    # 통합 파일 처리
    if info['all_results_files']:
        if keep_latest:
            # 최신 파일만 보존
            latest = max(info['all_results_files'], key=lambda f: f.stat().st_mtime)
            files_to_keep.append(latest)
            files_to_delete.extend([f for f in info['all_results_files'] if f != latest])
            print(f"\n통합 파일:")
            print(f"  ✓ 보존: {latest.name}")
            for f in files_to_delete:
                if f in info['all_results_files']:
                    print(f"  ✗ 삭제 예정: {f.name}")
        else:
            files_to_delete.extend(info['all_results_files'])
            print(f"\n통합 파일: 모두 삭제 예정")
    
    # 개별 파일 처리
    if info['individual_files']:
        files_to_delete.extend(info['individual_files'])
        print(f"\n개별 파일: 모두 삭제 예정 ({len(info['individual_files'])}개)")
    
    # 삭제할 파일 크기 계산
    if files_to_delete:
        total_delete_size = sum(f.stat().st_size for f in files_to_delete)
        total_delete_mb = total_delete_size / (1024 * 1024)
        total_delete_gb = total_delete_mb / 1024
        
        print(f"\n삭제 예정:")
        print(f"  • 파일 수: {len(files_to_delete)}개")
        print(f"  • 총 크기: {total_delete_mb:.1f} MB ({total_delete_gb:.2f} GB)")
        
        if not dry_run:
            print(f"\n삭제 중...")
            deleted = 0
            for f in files_to_delete:
                try:
                    f.unlink()
                    deleted += 1
                    if deleted % 5 == 0:
                        print(f"  삭제 중... ({deleted}/{len(files_to_delete)})")
                except Exception as e:
                    print(f"  ✗ 삭제 실패: {f.name} ({e})")
            
            print(f"\n✓ 삭제 완료: {deleted}/{len(files_to_delete)}개 파일")
            print(f"✓ 절약된 공간: {total_delete_mb:.1f} MB ({total_delete_gb:.2f} GB)")
        else:
            print(f"\n(DRY RUN: 실제로 삭제하지 않았습니다)")
    else:
        print("\n삭제할 파일이 없습니다.")
    
    # 보존할 파일
    if files_to_keep:
        print(f"\n보존된 파일:")
        for f in files_to_keep:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(
        description='이전 시뮬레이션 결과 파일 정리'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='삭제할 파일만 확인 (실제 삭제 안함)')
    parser.add_argument('--execute', action='store_true',
                       help='실제로 파일 삭제')
    parser.add_argument('--keep-latest', action='store_true', default=True,
                       help='최신 통합 파일 보존 (기본값: True)')
    parser.add_argument('--directory', type=str, default='results_comparison',
                       help='결과 디렉토리 (기본값: results_comparison)')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("⚠️  --dry-run 또는 --execute 옵션을 지정하세요.")
        print("\n사용법:")
        print("  python3 cleanup_old_results.py --dry-run    # 확인만")
        print("  python3 cleanup_old_results.py --execute    # 실제 삭제")
        return
    
    cleanup_files(
        directory=args.directory,
        dry_run=args.dry_run,
        keep_latest=args.keep_latest
    )

if __name__ == '__main__':
    main()

