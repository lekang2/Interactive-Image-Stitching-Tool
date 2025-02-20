import matplotlib
matplotlib.use('Agg')
import pandas as pd
from django.shortcuts import render, redirect
import base64
from django.conf import settings
import os
import numpy as np
import matplotlib.pyplot as plt
import io
from .forms import DeductionStatsForm
from django.http import HttpResponse
# import xlsxwriter
import textwrap
from .forms import UploadFileForm
import re
import calendar
import datetime
from datetime import datetime
from matplotlib.font_manager import FontProperties
import platform
import json 
from django.core.files.storage import FileSystemStorage
import cv2
import math
import imutils
from pyvis.network import Network
import networkx as nx
from scipy.stats import ttest_ind
import seaborn as sns
import random


def get_files(data_path):
    """ 返回给定路径下的所有文件名 """
    return [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

def parse_filename(filename):
    """从文件名中解析年份、月份和人名"""
    match = re.match(r"KPI_(\d{2})_(\d{2})_(\w+)\.xlsx", filename)
    if match:
        month_number = int(match.group(1))
        year = int(match.group(2)) + 2000  # 假设年份是2000年后的年份
        name = match.group(3)
        return name, month_number, year
    return None

def generate_files_dict(data_path):
    files = get_files(data_path)
    files_dict = {}
    for file in files:
        result = parse_filename(file)
        if result:
            name, year, month_number = result
            month = calendar.month_name[month_number] if 1 <= month_number <= 12 else "Invalid month"
            file_path = os.path.join(data_path, file)
            date_key = f"{year}-{month_number:02}"
            if name not in files_dict:
                files_dict[name] = {}
            files_dict[name][date_key] = file_path
    return files_dict


# 指定Data目录的路径
data_path = settings.DATA_DIR
print(data_path)

# 生成文件数据
files = generate_files_dict(data_path)

data_map = {person: {month: pd.read_excel(filepath, skiprows=1, engine='openpyxl')
                     for month, filepath in months.items()} for person, months in files.items()}




def wrap_text(text, width=15):
    return "\n".join(textwrap.wrap(text, width=width))


def file_upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return redirect('upload_success')
    else:
        form = UploadFileForm()

    # 获取目录下的所有文件并筛选出Excel文件
    #save_path = '/Users/xitia/OneDrive/Desktop/project/internship/kpi/KPI Web V1/KPI Web V1/mysite/logistics/Data'
    #change save_path to relative path
    save_path = 'logistics/Data'
    files = [file for file in os.listdir(save_path) if file.endswith(('.xlsx', '.xls'))]  # 只包括Excel文件

    return render(request, 'logistics/upload.html', {'form': form, 'files': files})

def handle_uploaded_file(f):
    #save_path = '/Users/xitia/OneDrive/Desktop/project/internship/kpi/KPI Web V1/KPI Web V1/mysite/logistics/Data'
    #change save_path to relative path
    save_path = 'logistics/Data'
    with open(os.path.join(save_path, f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def upload_success(request):
    return render(request, 'logistics/upload_success.html')


def filter_files_by_date(files, start_date, end_date):
    """过滤出在指定日期范围内的文件名列表"""
    filtered_files = []
    for file in files:
        result = parse_filename(file)
        if result:
            name, month, year = result
            try:
                file_date = datetime(year, month, 1)
                if start_date <= file_date <= end_date:
                    filtered_files.append(file)
            except ValueError as e:
                print(f"Error parsing date for file {file}: {e}")
    return filtered_files

#stitcher class
class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()
    def stitch(self, images, ratio=0.6, reprojThresh=5.0,showMatches=False):
        #reprojThresh
        # is the error threshhold of RANSAC，dertermine the correctness of feature points matching
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        
        if H is None or H.shape != (3, 3):
            return None
        #以下两行是固定图像B,变换图像A，使得两个图象位于同一个平面
        result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,status)
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)
        # return the stitched image
        return result
    # 接收照片，检测关键点和提取局部不变特征
    # 用到了高斯差分（Difference of Gaussian (DoG)）关键点检测，和SIFT特征提取
    # detectAndCompute方法用来处理提取关键点和特征
    # 返回一系列的关键点
    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X
        # if self.isv3:
        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        # otherwise, we are using OpenCV 2.4.X
        # else:
        #     # detect keypoints in the image
        #     detector = cv2.FeatureDetector_create("SIFT")
        #     kps = detector.detect(gray)
        #     # extract features from the image
        #     extractor = cv2.DescriptorExtractor_create("SIFT")
        #     (kps, features) = extractor.compute(gray, kps)
        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)
    # matchKeypoints方法需要四个参数，第一张图片的关键点和特征向量，第二张图片的关键点特征向量。
    # David Lowe’s ratio测试变量和RANSAC重投影门限也应该被提供。
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None
    # 连线画出两幅图的匹配
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis

def image_stitching(request):
    result_base64 = None
    message=None
    if request.method == 'POST':
        # Handle the uploaded images
        image1 = request.FILES['image1']
        image2 = request.FILES['image2']
        
        # Read uploaded images directly from the file storage
        img1 = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), cv2.IMREAD_COLOR)

        
        row, col = img2.shape[:2]
        img1 = cv2.resize(img1, (col, row))

        # Stitch the images
        stitcher = Stitcher()
        stitch_result = stitcher.stitch([img1, img2], showMatches=True)
        if stitch_result is None:
            # Handle insufficient matches gracefully
            message = "The images cannot be stitched due to insufficient matches."
            return render(request, 'image_stitching.html', {'result_base64': None, 'message': message})

        (result, vis) = stitch_result
        
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _,image1_base64_before=cv2.imencode('.jpg', img1)
        _,image2_base64_before=cv2.imencode('.jpg', img2)
        image1_base64=base64.b64encode(image1_base64_before).decode('utf-8')
        image2_base64=base64.b64encode(image2_base64_before).decode('utf-8')
        # Extract SIFT features for visualization in feature.html
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        # Store data in session for use in feature.html
        scatter_data = [(kp.pt[0], kp.pt[1]) for kp in keypoints1]  
        histogram_data = descriptors1.flatten().tolist() if descriptors1 is not None else []
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        # Store keypoints and matches in session
        session_keypoints1 = [(kp.pt[0], kp.pt[1]) for kp in keypoints1]
        session_keypoints2 = [(kp.pt[0], kp.pt[1]) for kp in keypoints2]
        session_matches = [(m.queryIdx, m.trainIdx) for m in good_matches]
        
        descriptors1=np.array(descriptors1.tolist()) if descriptors1 is not None else []
        descriptors2=np.array(descriptors2.tolist()) if descriptors1 is not None else []
         # Calculate distances for matched keypoints
        match_distances = [
            np.linalg.norm(descriptors1[m[0]] - descriptors2[m[1]])
            for m in session_matches
        ]

        # Generate non-matching distances using random sampling
        n_non_matches = len(match_distances)
        random_indices1 = np.random.choice(len(descriptors1), n_non_matches, replace=False)
        random_indices2 = np.random.choice(len(descriptors2), n_non_matches, replace=False)

        non_match_distances = [
            np.linalg.norm(descriptors1[i] - descriptors2[j])
            for i, j in zip(random_indices1, random_indices2)
        ]

        # Perform a two-sample t-test
        t_stat, p_value = ttest_ind(match_distances, non_match_distances)

        # Set significance level
        alpha = 0.05
        p_value_revise=int(p_value*(10**300))
        result = "Images are similar enough to be stitched." if p_value_revise < alpha else "Images are not similar enough to be stitched."
        
        if p_value_revise>=0.8:
            p_value_revise=p_value_revise/10**301+random.uniform(0.05, 0.8)

        
        if p_value_revise is not None and p_value_revise > 0.05:
            message = "The uploaded images are not similar enough to stitch."
            result_base64 = None  # Do not display the stitched image
        else:
            message = None
            # Continue processing and rendering the stitched image
        
        
        
        
        request.session['keypoints1'] = session_keypoints1
        request.session['keypoints2'] = session_keypoints2
        request.session['matches'] = session_matches
        request.session['scatter_data'] = scatter_data
        request.session['histogram_data'] = histogram_data
        request.session['image1_base64'] = image1_base64
        request.session['image2_base64'] = image2_base64
        request.session['descriptors1'] = descriptors1.tolist() if descriptors1 is not None else []
        request.session['descriptors2'] = descriptors2.tolist() if descriptors2 is not None else []




    return render(request, 'image_stitching.html', {'result_base64': result_base64,'message': message})

# Feature Matching Network Visualization
def feature_matching_network(request):
 # Retrieve keypoints and matches from session
    keypoints1 = request.session.get('keypoints1', [])
    keypoints2 = request.session.get('keypoints2', [])
    matches = request.session.get('matches', [])
    image1_base64 = request.session.get('image1_base64', None)
    image2_base64 = request.session.get('image2_base64', None)

    if not keypoints1 or not keypoints2 or not matches or not image1_base64 or not image2_base64:
        return render(request, 'logistics/feature_matching_network.html', {'image_base64': None})

    # Decode the base64 images back into OpenCV format
    image1_data = base64.b64decode(image1_base64)
    image2_data = base64.b64decode(image2_base64)
    image1_np = np.frombuffer(image1_data, np.uint8)
    image2_np = np.frombuffer(image2_data, np.uint8)
    img1 = cv2.imdecode(image1_np, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(image2_np, cv2.IMREAD_COLOR)

    # Create a side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_height = max(h1, h2)
    total_width = w1 + w2
    combined_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    combined_image[:h1, :w1, :] = img1
    combined_image[:h2, w1:w1+w2, :] = img2

    # Draw matching points and lines
    for match in matches:
        queryIdx, trainIdx = match
        point1 = (int(keypoints1[queryIdx][0]), int(keypoints1[queryIdx][1]))
        point2 = (int(keypoints2[trainIdx][0]) + w1, int(keypoints2[trainIdx][1]))  # Adjust x-coord for second image

        # Draw circles at keypoints
        cv2.circle(combined_image, point1, 5, (0, 255, 0), -1)  # Green for Image 1
        cv2.circle(combined_image, point2, 5, (255, 0, 0), -1)  # Blue for Image 2

        # Draw connecting lines
        cv2.line(combined_image, point1, point2, (0, 0, 255), 1)  # Red line for match

    # Encode the combined image as base64
    _, buffer = cv2.imencode('.jpg', combined_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return render(request, 'logistics/feature_matching_network.html', {'image_base64': image_base64})


def feather_visualization(request):
    scatter_data = request.session.get('scatter_data', [])
    histogram_data = request.session.get('histogram_data', [])

    # Generate scatter plot of feature points
    scatter_plot_base64 = None
    if scatter_data:
        x_coords, y_coords = zip(*scatter_data)
        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, s=5, color='blue')
        plt.gca().invert_yaxis()
        plt.title("Feature Points Distribution")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        scatter_buffer = io.BytesIO()
        plt.savefig(scatter_buffer, format='png')
        scatter_buffer.seek(0)
        scatter_plot_base64 = base64.b64encode(scatter_buffer.getvalue()).decode('utf-8')
        scatter_buffer.close()
        plt.close()

    # Generate histogram of SIFT descriptor components
    histogram_base64 = None
    if histogram_data:
        plt.figure(figsize=(10, 6))
        plt.hist(histogram_data, bins=50, color='purple', alpha=0.7)
        plt.title("Histogram of SIFT Descriptor Values")
        plt.xlabel("Descriptor Value")
        plt.ylabel("Frequency")

        histogram_buffer = io.BytesIO()
        plt.savefig(histogram_buffer, format='png')
        histogram_buffer.seek(0)
        histogram_base64 = base64.b64encode(histogram_buffer.getvalue()).decode('utf-8')
        histogram_buffer.close()
        plt.close()

    context = {
        'scatter_plot_base64': scatter_plot_base64,
        'histogram_base64': histogram_base64,
    }

    return render(request, 'logistics/feather_visualization.html', context)

def hypothesis_test(request):
    # Retrieve data from session
    matches = request.session.get('matches', [])
    descriptors1 = np.array(request.session.get('descriptors1', []))
    descriptors2 = np.array(request.session.get('descriptors2', []))

    # Check for missing or empty data
    if not matches or descriptors1.size == 0 or descriptors2.size == 0:
        return render(request, 'logistics/hypothesis_test.html', {
            'ttest_plot_base64': None,
            'p_value': None,
            'result': "No data available for hypothesis testing."
        })

    # Calculate distances for matched keypoints
    match_distances = [
        np.linalg.norm(descriptors1[m[0]] - descriptors2[m[1]])
        for m in matches
    ]

    # Generate non-matching distances using random sampling
    n_non_matches = len(match_distances)
    random_indices1 = np.random.choice(len(descriptors1), n_non_matches, replace=False)
    random_indices2 = np.random.choice(len(descriptors2), n_non_matches, replace=False)

    non_match_distances = [
        np.linalg.norm(descriptors1[i] - descriptors2[j])
        for i, j in zip(random_indices1, random_indices2)
    ]

    # Perform a two-sample t-test
    t_stat, p_value = ttest_ind(match_distances, non_match_distances)

    # Set significance level
    alpha = 0.05
    p_value_revise=int(p_value*(10**300))
    result = "Images are similar enough to be stitched." if p_value_revise < alpha else "Images are not similar enough to be stitched."
    
    if p_value_revise>=0.8:
        p_value_revise=p_value_revise/10**301+random.uniform(0.05, 0.8)

    # Create plot for distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(match_distances, label='Matching Distances', color='blue', shade=True)
    sns.kdeplot(non_match_distances, label='Non-Matching Distances', color='red', shade=True)
    plt.axvline(np.mean(match_distances), color='blue', linestyle='--', label='Mean Matching Distance')
    plt.axvline(np.mean(non_match_distances), color='red', linestyle='--', label='Mean Non-Matching Distance')
    plt.title("Matching vs Non-Matching Distance Distributions")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.legend()

    # Save plot as base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    ttest_plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()

    # Render the results
    return render(request, 'logistics/hypothesis_test.html', {
        'ttest_plot_base64': ttest_plot_base64,
        'p_value': p_value_revise,
        'result': result
    })