# from rest_framework import serializers
# from rest_framework.authentication import TokenAuthentication
# from rest_framework.authtoken.models import Token
# from rest_framework.authtoken.views import ObtainAuthToken
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from rest_framework import status


# from .models import Account

# class BasicUserSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Account
#         fields = (
#             'id', 'name', 'email', 'is_admin', 'phone_number', 'address')



# class CustomAuthToken(ObtainAuthToken):

#     def post(self, request, *args, **kwargs):
#         request.data['username'] = request.data['email']
#         serializer = self.serializer_class(data=request.data,
#                                            context={'request': request})
#         serializer.is_valid(raise_exception=True)
#         user = serializer.validated_data['user']

#         token, created = Token.objects.get_or_create(user=user)

#         account = BasicUserSerializer(user).data

#         data = {
#             'token': token.key,
#             'account': account
#             }

#         return Response(data, 200)



# class GetUserApiView(APIView):
#     class OutputSerializer(serializers.ModelSerializer):
#         class Meta:
#             model = Account
#             fields = ('id', 'full_name')
    
#     authentication_classes = [TokenAuthentication, ]

#     def get(self, request):
#         try:
#             if 'token' in request.GET:
#                 token = request.GET['token']
#                 user = Token.objects.get(key=token).user
#                 account = BasicUserSerializer(user).data

#                 data = {
#                     'token': token,
#                     'account': account
#                     }
                
#                 return Response(data, 200)
#             else:
#                 return Response({"message": "Token"}, status.HTTP_400_BAD_REQUEST)
#         except Exception as e:
#             return Response({'message': str(e)}, status.HTTP_400_BAD_REQUEST)
