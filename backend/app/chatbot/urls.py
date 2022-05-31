from django.urls import path, include

# from rest_framework import routers
from . import views

# router = routers.DefaultRouter()
# router.register('register', views.RegisterViewSet)


urlpatterns = [
    # path('', include(router.urls)),
    path("generate_dialog/", views.GenerateDialogView.as_view()),
    path("choose_model/", views.ChooseModelView.as_view()),
    # path('auth/', views.CustomAuthToken.as_view()),
    # path('getuser/', views.GetUserApiView.as_view(), name='getUserByToken')
]
