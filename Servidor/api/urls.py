from django.urls import path
from .views import CompanyView, PruebaView

urlpatterns = [
    path('companies/', CompanyView.as_view(), name='companies_list'),
    path('asistente/', PruebaView.as_view(), name='asistente'),
    path('companies/<int:id>', CompanyView.as_view(), name='companies_process')
]
