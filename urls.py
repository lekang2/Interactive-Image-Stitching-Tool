from django.urls import include, path
from django.contrib import admin

from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.image_stitching, name="image_stitching"),
    path("feather_visualization/", views.feather_visualization, name="feather_visualization"),
    path('feature_matching_network/', views.feature_matching_network, name='feature_matching_network'),
    path('hypothesis_test/', views.hypothesis_test, name='hypothesis_test'),
]