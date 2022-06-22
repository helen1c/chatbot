from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import *

class AccountAdmin(UserAdmin):
    model = Account
    list_display = (
        'email', 'name',
        )
    search_fields = ('email',)
    readonly_fields = ('date_joined', 'last_login')

    filter_horizontal = ()
    list_filter = ()
    fieldsets = ()
    ordering = ()
    add_fieldsets = (None, {
            'classes': ('wide',),
            'fields': (
                'email', 'password1', 'password2', 'name', 'username', 'is_admin', 'is_staff', 'is_superuser',
                'phone_number', 'address', 'is_active'),
        }),




admin.site.register(Account, AccountAdmin)