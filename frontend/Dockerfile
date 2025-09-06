FROM nginx:alpine

# Copy frontend files into Nginx
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
