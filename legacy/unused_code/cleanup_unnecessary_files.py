#!/usr/bin/env python3
"""
불필요한 파일 정리 스크립트

정리 대상:
1. __pycache__ 디렉토리
2. .DS_Store 파일
3. 중복/임시 파일들
4. 오래된 결과 파일들
"""

import argparse
from pathlib import Path
import shutil

def cleanup_pycache(dry_run=True):
    """__pycache__ 디렉토리 삭제"""
    print("\n" + "="*70)
    print("[1] __pycache__ 디렉토리 정리")
    print("="*70)
    
    pycache_dirs = list(Path('.').rglob('__pycache__'))
    
    if not pycache_dirs:
        print("✓ __pycache__ 디렉토리가 없습니다.")
        return 0
    
    total_size = 0
    for dir_path in pycache_dirs:
        size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
        total_size += size
        size_kb = size / 1024
        print(f"  • {dir_path} ({size_kb:.1f} KB)")
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n총 크기: {total_size_mb:.2f} MB")
    
    if not dry_run:
        for dir_path in pycache_dirs:
            shutil.rmtree(dir_path)
            print(f"  ✓ 삭제: {dir_path}")
        print(f"\n✓ {len(pycache_dirs)}개 디렉토리 삭제 완료")
    else:
        print(f"\n(DRY RUN: {len(pycache_dirs)}개 디렉토리 삭제 예정)")
    
    return len(pycache_dirs)

def cleanup_ds_store(dry_run=True):
    """macOS .DS_Store 파일 삭제"""
    print("\n" + "="*70)
    print("[2] .DS_Store 파일 정리")
    print("="*70)
    
    ds_store_files = list(Path('.').rglob('.DS_Store'))
    
    if not ds_store_files:
        print("✓ .DS_Store 파일이 없습니다.")
        return 0
    
    total_size = sum(f.stat().st_size for f in ds_store_files)
    total_size_kb = total_size / 1024
    
    for f in ds_store_files:
        print(f"  • {f}")
    
    print(f"\n총 크기: {total_size_kb:.2f} KB")
    
    if not dry_run:
        for f in ds_store_files:
            f.unlink()
            print(f"  ✓ 삭제: {f}")
        print(f"\n✓ {len(ds_store_files)}개 파일 삭제 완료")
    else:
        print(f"\n(DRY RUN: {len(ds_store_files)}개 파일 삭제 예정)")
    
    return len(ds_store_files)

def cleanup_duplicate_files(dry_run=True):
    """중복/임시 파일 정리"""
    print("\n" + "="*70)
    print("[3] 중복/임시 파일 정리")
    print("="*70)
    
    files_to_delete = []
    
    # 1. 중복된 bootstrap 파일
    if Path('regime_aware_bootstrap_improved.py').exists():
        if Path('regime_aware_bootstrap.py').exists():
            files_to_delete.append(Path('regime_aware_bootstrap.py'))
            print("  • regime_aware_bootstrap.py (improved 버전이 있음)")
    
    # 2. results_comparison의 중복 그래프 파일들
    results_dir = Path('results_comparison')
    if results_dir.exists():
        # improved1, improved2, improved3, test 버전들
        patterns = ['*improved*.png', '*improved*.pdf', '*test*.png', '*test*.pdf']
        for pattern in patterns:
            for f in results_dir.glob(pattern):
                files_to_delete.append(f)
                print(f"  • {f.name}")
    
    # 3. results/ 디렉토리의 오래된 JSON 파일들 (최신 1개만 보존)
    results_old_dir = Path('results')
    if results_old_dir.exists():
        json_files = sorted(results_old_dir.glob('*.json'), 
                           key=lambda x: x.stat().st_mtime, 
                           reverse=True)
        if len(json_files) > 1:
            # 최신 파일 제외하고 모두 삭제
            for f in json_files[1:]:
                files_to_delete.append(f)
                print(f"  • {f.name} (오래된 결과 파일)")
    
    if not files_to_delete:
        print("✓ 삭제할 중복 파일이 없습니다.")
        return 0
    
    total_size = sum(f.stat().st_size for f in files_to_delete)
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n총 {len(files_to_delete)}개 파일, {total_size_mb:.2f} MB")
    
    if not dry_run:
        deleted = 0
        for f in files_to_delete:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"  ✗ 삭제 실패: {f} ({e})")
        print(f"\n✓ {deleted}/{len(files_to_delete)}개 파일 삭제 완료")
    else:
        print(f"\n(DRY RUN: {len(files_to_delete)}개 파일 삭제 예정)")
    
    return len(files_to_delete)

def cleanup_test_files(dry_run=True, keep_important=True):
    """테스트/임시 스크립트 정리"""
    print("\n" + "="*70)
    print("[4] 테스트/임시 스크립트 정리")
    print("="*70)
    
    # 보존할 파일들
    keep_files = {
        'test_parameter_validation.py',  # 유용한 테스트
        'tests/',  # 테스트 디렉토리 전체
    }
    
    # 삭제 고려 대상
    test_files = [
        'test_phase2.py',
        'run_phase1_improvements.py',
        'run_with_phase2.py',
        'run_mn_regime_scan.py',
    ]
    
    files_to_delete = []
    for fname in test_files:
        fpath = Path(fname)
        if fpath.exists():
            # keep_important가 True면 중요한 파일은 건너뜀
            if keep_important and any(keep in str(fpath) for keep in keep_files):
                print(f"  ⊘ 보존: {fname} (중요한 파일)")
                continue
            files_to_delete.append(fpath)
            print(f"  • {fname}")
    
    if not files_to_delete:
        print("✓ 삭제할 테스트 파일이 없습니다.")
        return 0
    
    total_size = sum(f.stat().st_size for f in files_to_delete)
    total_size_kb = total_size / 1024
    
    print(f"\n총 {len(files_to_delete)}개 파일, {total_size_kb:.2f} KB")
    
    if not dry_run:
        deleted = 0
        for f in files_to_delete:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                print(f"  ✗ 삭제 실패: {f} ({e})")
        print(f"\n✓ {deleted}/{len(files_to_delete)}개 파일 삭제 완료")
    else:
        print(f"\n(DRY RUN: {len(files_to_delete)}개 파일 삭제 예정)")
    
    return len(files_to_delete)

def main():
    parser = argparse.ArgumentParser(
        description='불필요한 파일 정리'
    )
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='삭제할 파일만 확인 (기본값: True)')
    parser.add_argument('--execute', action='store_true',
                       help='실제로 파일 삭제')
    parser.add_argument('--keep-tests', action='store_true',
                       help='테스트 파일 보존')
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    if not dry_run:
        response = input("\n⚠️  실제로 파일을 삭제합니다. 계속하시겠습니까? (yes/no): ")
        if response.lower() != 'yes':
            print("취소되었습니다.")
            return
    
    print("="*70)
    print("불필요한 파일 정리")
    print("="*70)
    
    total_deleted = 0
    
    # 1. __pycache__
    total_deleted += cleanup_pycache(dry_run)
    
    # 2. .DS_Store
    total_deleted += cleanup_ds_store(dry_run)
    
    # 3. 중복 파일
    total_deleted += cleanup_duplicate_files(dry_run)
    
    # 4. 테스트 파일
    if not args.keep_tests:
        total_deleted += cleanup_test_files(dry_run, keep_important=True)
    
    print("\n" + "="*70)
    if dry_run:
        print(f"DRY RUN 완료: {total_deleted}개 항목 삭제 예정")
        print("\n실제 삭제하려면: python3 cleanup_unnecessary_files.py --execute")
    else:
        print(f"✓ 정리 완료: {total_deleted}개 항목 삭제됨")
    print("="*70)

if __name__ == '__main__':
    main()

