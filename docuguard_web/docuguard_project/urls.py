"""
URL configuration for docuguard_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from pii_detector.views import home_view, download_pdf
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
import os

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", home_view, name="home"),
    # Add path for downloading PDFs
    path("download/<str:filename>", download_pdf, name="download_pdf"),
    # Add explicit URL pattern for serving media files
    path('media/<path:path>', serve, {
        'document_root': os.path.join(settings.BASE_DIR, 'media')
    }),
]

# Keep this for compatibility
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
