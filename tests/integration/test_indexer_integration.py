"""
Integration tests for the cross-file indexer with real project scenarios.
"""

import os
import tempfile
import time
from pathlib import Path
import subprocess
import sys
import pytest

from astlib.indexer import ProjectIndex
from astlib.symbols import SymbolType


class TestIndexerIntegration:
    """Integration tests for ProjectIndex with complex project structures."""
    
    @pytest.fixture
    def django_like_project(self):
        """Create a Django-like project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create Django-like structure
            app = root / "myapp"
            app.mkdir()
            
            # __init__.py
            (app / "__init__.py").write_text("")
            
            # models.py
            (app / "models.py").write_text("""
from django.db import models
from django.contrib.auth.models import AbstractUser
from .managers import UserManager

class User(AbstractUser):
    '''Custom user model.'''
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    
    objects = UserManager()
    
    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def get_full_name(self):
        return f"{self.first_name} {self.last_name}".strip()

class Post(models.Model):
    '''Blog post model.'''
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title
""")
            
            # views.py
            (app / "views.py").write_text("""
from django.shortcuts import render, get_object_or_404
from django.views.generic import ListView, DetailView
from .models import Post, User
from .forms import PostForm

class PostListView(ListView):
    '''List all posts.'''
    model = Post
    template_name = 'posts/list.html'
    context_object_name = 'posts'
    paginate_by = 10
    
    def get_queryset(self):
        return super().get_queryset().select_related('author')

class PostDetailView(DetailView):
    '''Show post details.'''
    model = Post
    template_name = 'posts/detail.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['related_posts'] = Post.objects.filter(
            author=self.object.author
        ).exclude(pk=self.object.pk)[:5]
        return context

def create_post(request):
    '''Create a new post.'''
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()
            return redirect('post-detail', pk=post.pk)
    else:
        form = PostForm()
    
    return render(request, 'posts/create.html', {'form': form})
""")
            
            # forms.py
            (app / "forms.py").write_text("""
from django import forms
from .models import Post

class PostForm(forms.ModelForm):
    '''Form for creating/editing posts.'''
    
    class Meta:
        model = Post
        fields = ['title', 'content']
        widgets = {
            'content': forms.Textarea(attrs={'rows': 10}),
        }
    
    def clean_title(self):
        title = self.cleaned_data.get('title')
        if len(title) < 5:
            raise forms.ValidationError('Title must be at least 5 characters.')
        return title
""")
            
            # managers.py
            (app / "managers.py").write_text("""
from django.contrib.auth.models import UserManager as BaseUserManager

class UserManager(BaseUserManager):
    '''Custom user manager.'''
    
    def get_by_natural_key(self, username):
        return self.get(username__iexact=username)
    
    def active_users(self):
        return self.filter(is_active=True)
""")
            
            # urls.py
            (app / "urls.py").write_text("""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.PostListView.as_view(), name='post-list'),
    path('post/<int:pk>/', views.PostDetailView.as_view(), name='post-detail'),
    path('post/new/', views.create_post, name='post-create'),
]
""")
            
            # tests.py
            (app / "tests.py").write_text("""
from django.test import TestCase
from django.contrib.auth import get_user_model
from .models import Post

User = get_user_model()

class PostModelTest(TestCase):
    '''Test Post model.'''
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
    
    def test_create_post(self):
        post = Post.objects.create(
            title='Test Post',
            content='Test content',
            author=self.user
        )
        self.assertEqual(str(post), 'Test Post')
        self.assertEqual(post.author, self.user)
""")
            
            yield root
    
    @pytest.fixture
    def flask_like_project(self):
        """Create a Flask-like project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create Flask app structure
            (root / "app.py").write_text("""
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import config

app = Flask(__name__)
app.config.from_object(config.Config)
db = SQLAlchemy(app)

# Import models after db initialization
from models import User, Post

@app.route('/')
def index():
    '''Home page showing recent posts.'''
    posts = Post.query.order_by(Post.created_at.desc()).limit(10).all()
    return render_template('index.html', posts=posts)

@app.route('/post/<int:post_id>')
def post_detail(post_id):
    '''Show post details.'''
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', post=post)

@app.route('/user/<username>')
def user_profile(username):
    '''Show user profile.'''
    user = User.query.filter_by(username=username).first_or_404()
    return render_template('profile.html', user=user)

if __name__ == '__main__':
    app.run(debug=True)
""")
            
            # models.py
            (root / "models.py").write_text("""
from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    '''User model.'''
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    posts = db.relationship('Post', backref='author', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Post(db.Model):
    '''Blog post model.'''
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Post {self.title}>'
""")
            
            # config.py
            (root / "config.py").write_text("""
import os

class Config:
    '''Flask configuration.'''
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
""")
            
            # Create utils package
            utils = root / "utils"
            utils.mkdir()
            (utils / "__init__.py").write_text("")
            
            (utils / "helpers.py").write_text("""
from functools import wraps
from flask import g, redirect, url_for

def login_required(f):
    '''Decorator to require login.'''
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def format_datetime(dt):
    '''Format datetime for display.'''
    return dt.strftime('%B %d, %Y at %I:%M %p')
""")
            
            yield root
    
    def test_django_project_indexing(self, django_like_project):
        """Test indexing a Django-like project."""
        index = ProjectIndex()
        stats = index.build_index(django_like_project)
        
        # Check basic stats
        assert stats.indexed_files >= 6
        assert stats.total_symbols > 20
        
        # Test finding Django models
        user_def = index.find_definition("User")
        assert user_def is not None
        assert user_def[0].name == "models.py"
        
        post_def = index.find_definition("Post")
        assert post_def is not None
        
        # Test finding views
        list_view = index.find_definition("PostListView")
        assert list_view is not None
        assert list_view[0].name == "views.py"
        
        # Test finding methods
        get_full_name = index.get_symbol("User.get_full_name")
        assert get_full_name is not None
        assert get_full_name.type == SymbolType.METHOD
        
        # Test cross-file imports
        views_imports = index.get_file_imports(django_like_project / "myapp" / "views.py")
        assert views_imports is not None
        assert ".models" in views_imports.imported_modules
        assert ".forms" in views_imports.imported_modules
    
    def test_flask_project_indexing(self, flask_like_project):
        """Test indexing a Flask-like project."""
        index = ProjectIndex()
        stats = index.build_index(flask_like_project)
        
        # Check stats
        assert stats.indexed_files >= 4
        assert stats.total_symbols > 15
        
        # Test finding Flask routes
        index_route = index.find_definition("index")
        assert index_route is not None
        
        # Test finding models
        user_model = index.find_definition("User")
        assert user_model is not None
        assert user_model[0].name == "models.py"
        
        # Test finding decorators
        login_required = index.find_definition("login_required")
        assert login_required is not None
        assert login_required[0].name == "helpers.py"
        
        # Test import graph
        graph = index.get_import_graph()
        assert "app" in graph.nodes
        assert "models" in graph.nodes
        assert "config" in graph.nodes
        
        # Check circular import potential (app imports from models, models imports from app)
        app_node = graph.nodes["app"]
        models_node = graph.nodes["models"]
        assert "models" in app_node.imports
        assert "app" in models_node.imports
    
    def test_cross_file_symbol_resolution(self, django_like_project):
        """Test resolving symbols across files."""
        index = ProjectIndex()
        index.build_index(django_like_project)
        
        # Find PostForm usage in views.py
        views_file = django_like_project / "myapp" / "views.py"
        form_def = index.find_definition("PostForm", from_file=views_file)
        assert form_def is not None
        assert form_def[0].name == "forms.py"
        
        # Find User model usage in tests.py
        tests_file = django_like_project / "myapp" / "tests.py"
        user_def = index.find_definition("User", from_file=tests_file)
        assert user_def is not None
    
    def test_incremental_refresh(self, flask_like_project):
        """Test incremental refresh functionality."""
        index = ProjectIndex()
        initial_stats = index.build_index(flask_like_project)
        
        # Add a new route to app.py
        app_file = flask_like_project / "app.py"
        content = app_file.read_text()
        new_content = content.replace(
            "if __name__ == '__main__':",
            """@app.route('/api/posts')
def api_posts():
    '''API endpoint for posts.'''
    posts = Post.query.all()
    return jsonify([p.to_dict() for p in posts])

if __name__ == '__main__':"""
        )
        
        # Wait to ensure mtime difference
        time.sleep(0.1)
        app_file.write_text(new_content)
        
        # Refresh
        refresh_stats = index.refresh()
        assert refresh_stats.indexed_files == 1
        
        # Check new function is indexed
        api_def = index.find_definition("api_posts")
        assert api_def is not None
    
    def test_find_all_references(self, django_like_project):
        """Test finding all references to a symbol."""
        index = ProjectIndex()
        index.build_index(django_like_project)
        
        # Find all references to Post model
        post_refs = index.find_references("Post")
        assert len(post_refs) >= 2  # At least in views.py and forms.py
        
        ref_files = {ref[0].name for ref in post_refs}
        assert "views.py" in ref_files
        assert "forms.py" in ref_files
    
    def test_symbol_hierarchy(self, django_like_project):
        """Test symbol parent-child relationships."""
        index = ProjectIndex()
        index.build_index(django_like_project)
        
        # Get User class
        user_class = index.get_symbol("User")
        assert user_class is not None
        assert user_class.type == SymbolType.CLASS
        
        # Get Meta class inside User
        meta_class = index.get_symbol("User.Meta")
        assert meta_class is not None
        assert meta_class.type == SymbolType.CLASS
        assert meta_class.parent.name == "User"
        
        # Get method inside class
        method = index.get_symbol("PostListView.get_queryset")
        assert method is not None
        assert method.type == SymbolType.METHOD
        assert method.parent.name == "PostListView"
    
    def test_performance_real_project(self, django_like_project):
        """Test performance on a more realistic project size."""
        # Add more files to simulate larger project
        for i in range(20):
            module_dir = django_like_project / f"module{i}"
            module_dir.mkdir()
            (module_dir / "__init__.py").write_text("")
            
            (module_dir / f"models{i}.py").write_text(f"""
class Model{i}:
    '''Model {i}'''
    def method_{i}(self):
        pass
    
    @property
    def prop_{i}(self):
        return f"Property {i}"

def function_{i}():
    '''Function {i}'''
    pass
""")
            
            (module_dir / f"views{i}.py").write_text(f"""
from .models{i} import Model{i}, function_{i}

class View{i}:
    '''View {i}'''
    model = Model{i}
    
    def get(self, request):
        return function_{i}()
""")
        
        # Index with timing
        index = ProjectIndex(max_workers=4)
        start_time = time.perf_counter()
        stats = index.build_index(django_like_project)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Should handle 40+ files efficiently
        assert stats.indexed_files >= 46
        assert stats.total_symbols > 200
        assert elapsed_ms < 3000  # Less than 3 seconds
    
    def test_cli_integration(self, flask_like_project):
        """Test CLI commands with the indexer."""
        # Test index command
        result = subprocess.run(
            [sys.executable, "-m", "astlib.cli", "index", "-p", str(flask_like_project)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Indexing complete" in result.stdout
        
        # Test find-symbol command
        result = subprocess.run(
            [sys.executable, "-m", "astlib.cli", "find-symbol", "User", "-p", str(flask_like_project)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "models.py" in result.stdout
        
        # Test list-symbols command
        result = subprocess.run(
            [sys.executable, "-m", "astlib.cli", "list-symbols", "-p", str(flask_like_project), "-t", "class"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "User" in result.stdout
        assert "Post" in result.stdout
        assert "Config" in result.stdout
        
        # Test show-imports command
        result = subprocess.run(
            [sys.executable, "-m", "astlib.cli", "show-imports", "-p", str(flask_like_project)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Import graph summary" in result.stdout