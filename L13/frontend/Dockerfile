# Minimal frontend: nginx serving static files
# In production you might build a React/Vue app and copy the dist/ here
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
