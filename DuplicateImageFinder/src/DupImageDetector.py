#!/usr/bin/env python3
"""
Duplicate Image Finder - A tool to find duplicate images on local drives
Based on visual content similarity using perceptual hashing
"""

import os
import sys
import argparse
import hashlib
import sqlite3
import json
import csv
import threading
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Ok so this isn't ideal the error message is a little confusing because we are doing two packages at once
#try:
#    from PIL import Image, ExifTags
#    import imagehash
#except ImportError:
#   print("Required packages missing. Install with:")
#   print("pip install Pillow imagehash")
#    sys.exit(1)

try:
    from PIL import Image, ExifTags
except ImportError:
    print("Required packages missing. Install with:")
    print("pip install Pillow")
    sys.exit(1)

try:
    import imagehash
except ImportError:
    print("Required packages missing. Install with:")
    print("pip install imagehash")
    #Note that's a pretty heavy set of dependencies, numpy, pywavelets, scipy
    sys.exit(1)



@dataclass
class ImageInfo:
    """Container for image metadata"""
    path: str
    size: int
    modified_time: float
    dimensions: Tuple[int, int]
    file_hash: str
    perceptual_hash: str
    exif_hash: Optional[str] = None

class DuplicateFinder:
    """Main class for finding duplicate images"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache_db = config.get('cache_db', 'image_cache.db')
        self.similarity_threshold = config.get('threshold', 95)
        self.min_size = config.get('min_size', 1024)  # 1KB default
        self.max_size = config.get('max_size', 100 * 1024 * 1024)  # 100MB default
        self.thread_count = config.get('threads', min(32, os.cpu_count() * 2))
        self.verbose = config.get('verbose', 0)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'duplicates_found': 0,
            'space_wasted': 0,
            'errors': 0
        }
        
        # Thread-safe progress tracking
        self.progress_lock = threading.Lock()
        
        # Initialize database
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        try:
            self.conn = sqlite3.connect(self.cache_db, check_same_thread=False)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS image_cache (
                    path TEXT PRIMARY KEY,
                    file_hash TEXT,
                    perceptual_hash TEXT,
                    size INTEGER,
                    modified_time REAL,
                    width INTEGER,
                    height INTEGER,
                    exif_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON image_cache(file_hash)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON image_cache(perceptual_hash)')
            self.conn.commit()
        except Exception as e:
            self._log(f"Warning: Could not initialize cache database: {e}", 1)
            self.conn = None
    
    def _log(self, message: str, level: int = 0):
        """Thread-safe logging with verbosity levels"""
        if level <= self.verbose:
            print(f"[{threading.current_thread().name}] {message}")
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file"""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self._log(f"Error hashing {filepath}: {e}", 1)
            return ""
    
    def _get_exif_hash(self, image: Image.Image) -> Optional[str]:
        """Extract and hash EXIF data for additional comparison"""
        try:
            exif = image._getexif()
            if exif:
                # Focus on key EXIF fields that indicate same photo
                key_fields = [
                    'DateTime', 'DateTimeOriginal', 'DateTimeDigitized',
                    'Make', 'Model', 'ExposureTime', 'FNumber', 'ISOSpeedRatings'
                ]
                exif_data = {}
                for field in key_fields:
                    for tag, value in exif.items():
                        if ExifTags.TAGS.get(tag) == field:
                            exif_data[field] = str(value)
                            break
                
                if exif_data:
                    exif_str = json.dumps(exif_data, sort_keys=True)
                    return hashlib.md5(exif_str.encode()).hexdigest()
        except Exception:
            pass
        return None
    
    def _process_image(self, filepath: str) -> Optional[ImageInfo]:
        """Process a single image file and extract metadata"""
        try:
            # Check file size constraints
            file_size = os.path.getsize(filepath)
            if file_size < self.min_size or file_size > self.max_size:
                with self.progress_lock:
                    self.stats['files_skipped'] += 1
                return None
            
            # Check cache first
            if self.conn:
                modified_time = os.path.getmtime(filepath)
                cursor = self.conn.execute(
                    'SELECT * FROM image_cache WHERE path = ? AND modified_time = ?',
                    (filepath, modified_time)
                )
                cached = cursor.fetchone()
                if cached:
                    self._log(f"Using cached data for {filepath}", 2)
                    return ImageInfo(
                        path=cached[0],
                        file_hash=cached[1],
                        perceptual_hash=cached[2],
                        size=cached[3],
                        modified_time=cached[4],
                        dimensions=(cached[5], cached[6]),
                        exif_hash=cached[7]
                    )
            
            # Process new image
            with Image.open(filepath) as img:
                # Convert to RGB if necessary for consistent hashing
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate hashes
                file_hash = self._get_file_hash(filepath)
                perceptual_hash = str(imagehash.phash(img, hash_size=16))
                exif_hash = self._get_exif_hash(img)
                
                image_info = ImageInfo(
                    path=filepath,
                    size=file_size,
                    modified_time=os.path.getmtime(filepath),
                    dimensions=img.size,
                    file_hash=file_hash,
                    perceptual_hash=perceptual_hash,
                    exif_hash=exif_hash
                )
                
                # Cache the result
                if self.conn:
                    try:
                        self.conn.execute('''
                            INSERT OR REPLACE INTO image_cache 
                            (path, file_hash, perceptual_hash, size, modified_time, width, height, exif_hash)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            filepath, file_hash, perceptual_hash, file_size,
                            image_info.modified_time, img.size[0], img.size[1], exif_hash
                        ))
                        self.conn.commit()
                    except Exception as e:
                        self._log(f"Cache write error: {e}", 1)
                
                with self.progress_lock:
                    self.stats['files_processed'] += 1
                
                self._log(f"Processed: {filepath}", 2)
                return image_info
                
        except Exception as e:
            self._log(f"Error processing {filepath}: {e}", 1)
            with self.progress_lock:
                self.stats['errors'] += 1
            return None
    
    def _find_image_files(self, directories: List[str], recursive: bool = True) -> List[str]:
        """Find all image files in specified directories"""
        image_files = []
        
        for directory in directories:
            if not os.path.exists(directory):
                self._log(f"Directory not found: {directory}", 0)
                continue
                
            path_obj = Path(directory)
            
            if recursive:
                pattern = '**/*'
            else:
                pattern = '*'
            
            for file_path in path_obj.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity percentage between two perceptual hashes"""
        if len(hash1) != len(hash2):
            return 0.0
        
        # Convert hex strings to integers for XOR comparison
        try:
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            
            # XOR and count different bits
            diff_bits = bin(int1 ^ int2).count('1')
            total_bits = len(hash1) * 4  # 4 bits per hex character
            
            similarity = ((total_bits - diff_bits) / total_bits) * 100
            return similarity
        except ValueError:
            return 0.0
    
    def _group_duplicates(self, images: List[ImageInfo]) -> Dict[str, List[ImageInfo]]:
        """Group images by similarity"""
        duplicate_groups = defaultdict(list)
        processed_hashes = set()
        
        # First, group by exact file hash (identical files)
        file_hash_groups = defaultdict(list)
        for img in images:
            if img.file_hash:
                file_hash_groups[img.file_hash].append(img)
        
        # Add identical file groups
        for file_hash, group in file_hash_groups.items():
            if len(group) > 1:
                duplicate_groups[f"identical_{file_hash}"] = group
                for img in group:
                    processed_hashes.add(img.perceptual_hash)
        
        # Group by perceptual hash similarity
        remaining_images = [img for img in images if img.perceptual_hash not in processed_hashes]
        
        for i, img1 in enumerate(remaining_images):
            if img1.perceptual_hash in processed_hashes:
                continue
                
            similar_group = [img1]
            processed_hashes.add(img1.perceptual_hash)
            
            for img2 in remaining_images[i+1:]:
                if img2.perceptual_hash in processed_hashes:
                    continue
                    
                similarity = self._calculate_similarity(img1.perceptual_hash, img2.perceptual_hash)
                
                # Also check EXIF similarity for additional validation
                exif_match = (img1.exif_hash and img2.exif_hash and 
                            img1.exif_hash == img2.exif_hash)
                
                if similarity >= self.similarity_threshold or exif_match:
                    similar_group.append(img2)
                    processed_hashes.add(img2.perceptual_hash)
            
            if len(similar_group) > 1:
                duplicate_groups[f"similar_{img1.perceptual_hash}"] = similar_group
        
        return dict(duplicate_groups)
    
    def find_duplicates(self, directories: List[str], recursive: bool = True) -> Dict[str, List[ImageInfo]]:
        """Main method to find duplicate images"""
        self._log(f"Starting duplicate search in: {directories}")
        self._log(f"Recursive: {recursive}, Threshold: {self.similarity_threshold}%")
        
        # Find all image files
        image_files = self._find_image_files(directories, recursive)
        total_files = len(image_files)
        
        if total_files == 0:
            self._log("No image files found!")
            return {}
        
        self._log(f"Found {total_files} image files to process")
        
        # Process images with progress tracking
        images = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            future_to_file = {executor.submit(self._process_image, filepath): filepath 
                            for filepath in image_files}
            
            for i, future in enumerate(as_completed(future_to_file)):
                if i % 50 == 0:  # Progress update every 50 files
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        rate = (i + 1) / elapsed
                        eta = (total_files - i - 1) / rate if rate > 0 else 0
                        self._log(f"Progress: {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%) "
                               f"ETA: {eta:.0f}s", 0)
                
                result = future.result()
                if result:
                    images.append(result)
        
        self._log(f"Processing complete. Processed: {len(images)} images")
        
        # Find duplicates
        duplicate_groups = self._group_duplicates(images)
        
        # Update statistics
        self.stats['duplicates_found'] = sum(len(group) - 1 for group in duplicate_groups.values())
        self.stats['space_wasted'] = sum(
            sum(img.size for img in group[1:])  # All but the first (kept) image
            for group in duplicate_groups.values()
        )
        
        return duplicate_groups
    
    def generate_report(self, duplicate_groups: Dict[str, List[ImageInfo]], 
                       output_format: str = 'console', output_file: str = None):
        """Generate report in specified format"""
        
        if output_format == 'console':
            self._print_console_report(duplicate_groups)
        elif output_format == 'csv':
            self._generate_csv_report(duplicate_groups, output_file or 'duplicates.csv')
        elif output_format == 'json':
            self._generate_json_report(duplicate_groups, output_file or 'duplicates.json')
        elif output_format == 'html':
            self._generate_html_report(duplicate_groups, output_file or 'duplicates.html')
    
    def _print_console_report(self, duplicate_groups: Dict[str, List[ImageInfo]]):
        """Print report to console"""
        print("\n" + "="*80)
        print("DUPLICATE IMAGE FINDER REPORT")
        print("="*80)
        
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files skipped: {self.stats['files_skipped']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Duplicate groups found: {len(duplicate_groups)}")
        print(f"Duplicate images: {self.stats['duplicates_found']}")
        print(f"Potential space savings: {self.stats['space_wasted'] / (1024*1024):.2f} MB")
        
        if duplicate_groups:
            print("\nDUPLICATE GROUPS:")
            print("-" * 40)
            
            for group_id, images in duplicate_groups.items():
                print(f"\nGroup: {group_id}")
                print(f"Images: {len(images)}")
                
                # Sort by size (largest first) and modification time
                sorted_images = sorted(images, key=lambda x: (-x.size, -x.modified_time))
                
                for i, img in enumerate(sorted_images):
                    marker = "[KEEP]" if i == 0 else "[DELETE]"
                    mod_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                          time.localtime(img.modified_time))
                    print(f"  {marker} {img.path}")
                    print(f"    Size: {img.size:,} bytes, Modified: {mod_time}")
                    print(f"    Dimensions: {img.dimensions[0]}x{img.dimensions[1]}")
    
    def _generate_csv_report(self, duplicate_groups: Dict[str, List[ImageInfo]], filename: str):
        """Generate CSV report"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Group', 'Action', 'Path', 'Size', 'Dimensions', 'Modified', 'File_Hash', 'Perceptual_Hash'])
            
            for group_id, images in duplicate_groups.items():
                sorted_images = sorted(images, key=lambda x: (-x.size, -x.modified_time))
                for i, img in enumerate(sorted_images):
                    action = 'KEEP' if i == 0 else 'DELETE'
                    mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(img.modified_time))
                    writer.writerow([
                        group_id, action, img.path, img.size,
                        f"{img.dimensions[0]}x{img.dimensions[1]}", mod_time,
                        img.file_hash, img.perceptual_hash
                    ])
        
        print(f"CSV report saved to: {filename}")
    
    def _generate_json_report(self, duplicate_groups: Dict[str, List[ImageInfo]], filename: str):
        """Generate JSON report"""
        report_data = {
            'statistics': self.stats,
            'duplicate_groups': {}
        }
        
        for group_id, images in duplicate_groups.items():
            sorted_images = sorted(images, key=lambda x: (-x.size, -x.modified_time))
            report_data['duplicate_groups'][group_id] = [
                {
                    'path': img.path,
                    'size': img.size,
                    'dimensions': img.dimensions,
                    'modified_time': img.modified_time,
                    'file_hash': img.file_hash,
                    'perceptual_hash': img.perceptual_hash,
                    'exif_hash': img.exif_hash,
                    'action': 'KEEP' if i == 0 else 'DELETE'
                }
                for i, img in enumerate(sorted_images)
            ]
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(report_data, jsonfile, indent=2)
        
        print(f"JSON report saved to: {filename}")
    
    def _generate_html_report(self, duplicate_groups: Dict[str, List[ImageInfo]], filename: str):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Duplicate Image Finder Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .stats {{ background: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
                .group {{ border: 1px solid #ccc; margin: 10px 0; padding: 10px; }}
                .keep {{ background: #e8f5e8; }}
                .delete {{ background: #ffe8e8; }}
                .image-info {{ margin: 5px 0; font-size: 14px; }}
            </style>
        </head>
        <body>
            <h1>Duplicate Image Finder Report</h1>
            
            <div class="stats">
                <h2>Statistics</h2>
                <p>Files processed: {self.stats['files_processed']}</p>
                <p>Files skipped: {self.stats['files_skipped']}</p>
                <p>Errors: {self.stats['errors']}</p>
                <p>Duplicate groups: {len(duplicate_groups)}</p>
                <p>Duplicate images: {self.stats['duplicates_found']}</p>
                <p>Potential space savings: {self.stats['space_wasted'] / (1024*1024):.2f} MB</p>
            </div>
        """
        
        for group_id, images in duplicate_groups.items():
            html_content += f'<div class="group"><h3>Group: {group_id}</h3>'
            sorted_images = sorted(images, key=lambda x: (-x.size, -x.modified_time))
            
            for i, img in enumerate(sorted_images):
                css_class = 'keep' if i == 0 else 'delete'
                action = 'KEEP' if i == 0 else 'DELETE'
                mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(img.modified_time))
                
                html_content += f'''
                <div class="image-info {css_class}">
                    <strong>[{action}]</strong> {img.path}<br>
                    Size: {img.size:,} bytes | Dimensions: {img.dimensions[0]}x{img.dimensions[1]} | Modified: {mod_time}
                </div>
                '''
            
            html_content += '</div>'
        
        html_content += '</body></html>'
        
        with open(filename, 'w', encoding='utf-8') as htmlfile:
            htmlfile.write(html_content)
        
        print(f"HTML report saved to: {filename}")
    
    def auto_delete_duplicates(self, duplicate_groups: Dict[str, List[ImageInfo]], 
                              rule: str = 'newest', dry_run: bool = True):
        """Automatically delete duplicates based on specified rule"""
        deleted_files = []
        space_freed = 0
        
        for group_id, images in duplicate_groups.items():
            if len(images) <= 1:
                continue
            
            # Sort based on rule
            if rule == 'newest':
                sorted_images = sorted(images, key=lambda x: -x.modified_time)
            elif rule == 'oldest':
                sorted_images = sorted(images, key=lambda x: x.modified_time)
            elif rule == 'largest':
                sorted_images = sorted(images, key=lambda x: -x.size)
            elif rule == 'smallest':
                sorted_images = sorted(images, key=lambda x: x.size)
            else:
                sorted_images = images
            
            # Keep first, delete rest
            to_delete = sorted_images[1:]
            
            for img in to_delete:
                if dry_run:
                    print(f"Would delete: {img.path}")
                else:
                    try:
                        os.remove(img.path)
                        deleted_files.append(img.path)
                        space_freed += img.size
                        print(f"Deleted: {img.path}")
                    except Exception as e:
                        print(f"Error deleting {img.path}: {e}")
        
        if dry_run:
            print(f"\nDry run complete. Would delete {len(deleted_files)} files")
            print(f"Would free {space_freed / (1024*1024):.2f} MB")
        else:
            print(f"\nDeleted {len(deleted_files)} duplicate files")
            print(f"Freed {space_freed / (1024*1024):.2f} MB")
        
        return deleted_files, space_freed

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Find duplicate images on local drives')
    parser.add_argument('directories', nargs='+', help='Directories to scan')
    parser.add_argument('-r', '--recursive', action='store_true', default=True,
                       help='Scan subdirectories (default: True)')
    parser.add_argument('-t', '--threshold', type=int, default=95,
                       help='Similarity threshold percentage (default: 95)')
    parser.add_argument('-f', '--formats', default='jpg,jpeg,png,gif,bmp,tiff,tif,webp',
                       help='File formats to include (default: jpg,jpeg,png,gif,bmp,tiff,tif,webp)')
    parser.add_argument('--min-size', type=int, default=1024,
                       help='Minimum file size in bytes (default: 1024)')
    parser.add_argument('--max-size', type=int, default=100*1024*1024,
                       help='Maximum file size in bytes (default: 100MB)')
    parser.add_argument('-o', '--output', choices=['console', 'csv', 'json', 'html'],
                       default='console', help='Output format (default: console)')
    parser.add_argument('--output-file', help='Output file path (auto-generated if not specified)')
    parser.add_argument('--cache-db', default='image_cache.db',
                       help='Cache database location (default: image_cache.db)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show results without taking action')
    parser.add_argument('--auto-delete', choices=['newest', 'oldest', 'largest', 'smallest'],
                       help='Automatic deletion rule')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase verbosity level')
    parser.add_argument('--threads', type=int,
                       help='Number of processing threads (default: auto)')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'threshold': args.threshold,
        'min_size': args.min_size,
        'max_size': args.max_size,
        'cache_db': args.cache_db,
        'verbose': args.verbose,
        'threads': args.threads or min(32, os.cpu_count() * 2)
    }
    
    # Create finder and run
    finder = DuplicateFinder(config)
    
    try:
        duplicate_groups = finder.find_duplicates(args.directories, args.recursive)
        
        if duplicate_groups:
            finder.generate_report(duplicate_groups, args.output, args.output_file)
            # remove autodelete option
            # if args.auto_delete:
            #     finder.auto_delete_duplicates(duplicate_groups, args.auto_delete, args.dry_run)
        else:
            print("No duplicate images found!")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose > 0:
            import traceback
            traceback.print_exc()
    finally:
        if hasattr(finder, 'conn') and finder.conn:
            finder.conn.close()

if __name__ == '__main__':
    main()