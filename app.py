"""
Streamlit Web Application for Face Authentication System.
Provides a user-friendly interface for user registration and authentication.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os

from face_authentication import FaceAuthenticationSystem
from utils import preprocess_image


# Page configuration
st.set_page_config(
    page_title="Face Authentication System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_auth_system():
    """Load and cache the authentication system."""
    # Check for model file
    model_path = 'best_model.h5'
    if not os.path.exists(model_path):
        model_path = 'hyperspectral_face_recognition_model.keras'
    if not os.path.exists(model_path):
        st.warning("⚠️ No trained model found. Using dummy model for demonstration.")
        model_path = 'nonexistent_model.h5'
    
    auth_system = FaceAuthenticationSystem(
        model_path=model_path,
        database_path='face_database.pkl',
        use_gabor=True,
        similarity_threshold=0.6
    )
    
    return auth_system


def main():
    """Main application function."""
    
    # Title
    st.title("🔐 Face Authentication System")
    st.markdown("### Hyperspectral Face Recognition with Gabor Transform")
    
    # Load authentication system
    try:
        auth_system = load_auth_system()
    except Exception as e:
        st.error(f"Error loading authentication system: {str(e)}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Threshold control
        threshold = st.slider(
            "Authentication Threshold",
            min_value=0.0,
            max_value=1.0,
            value=auth_system.similarity_threshold,
            step=0.05,
            help="Minimum similarity score required for authentication"
        )
        auth_system.set_threshold(threshold)
        
        st.markdown("---")
        
        # System statistics
        st.header("📊 System Statistics")
        stats = auth_system.get_statistics()
        st.metric("Registered Users", stats['total_users'])
        st.metric("Feature Dimension", stats['feature_dimension'])
        st.metric("Threshold", f"{stats['similarity_threshold']:.2f}")
        
        if st.checkbox("Show Gabor Settings"):
            st.write(f"**Use Gabor Transform:** {stats['use_gabor']}")
            st.write(f"**Model Path:** {os.path.basename(stats['model_path'])}")
        
        st.markdown("---")
        
        # Database management
        st.header("🗄️ Database")
        
        if st.button("📋 View All Users"):
            users = auth_system.list_users()
            if users:
                st.write(f"**Total Users:** {len(users)}")
                for user in users:
                    with st.expander(f"👤 {user['user_id']}"):
                        st.write(f"**Samples:** {user['num_samples']}")
                        st.write(f"**Registered:** {user['registration_date'][:10]}")
                        if user.get('metadata'):
                            st.write("**Metadata:**", user['metadata'])
            else:
                st.info("No users registered yet")
        
        if st.button("💾 Export Database"):
            try:
                export_path = "database_export.json"
                auth_system.export_database(export_path, format='json')
                st.success(f"✅ Database exported to {export_path}")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    # Main content area - tabs
    tab1, tab2, tab3 = st.tabs(["👤 User Registration", "🔍 Authentication", "✅ Verification"])
    
    # Tab 1: User Registration
    with tab1:
        st.header("User Registration")
        st.markdown("Register a new user by uploading face images")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # User ID input
            user_id = st.text_input(
                "User ID",
                placeholder="Enter unique user ID (e.g., john_doe)",
                help="Choose a unique identifier for the user"
            )
            
            # Metadata
            st.subheader("Optional Metadata")
            user_name = st.text_input("Full Name", placeholder="John Doe")
            user_email = st.text_input("Email", placeholder="john@example.com")
            user_dept = st.text_input("Department", placeholder="Engineering")
            
            # Image upload
            st.subheader("Upload Face Images")
            st.info("💡 Upload 2-5 images of the same person for better accuracy")
            
            uploaded_files = st.file_uploader(
                "Choose face images",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                accept_multiple_files=True,
                key="registration_upload"
            )
        
        with col2:
            if uploaded_files:
                st.subheader(f"Uploaded Images ({len(uploaded_files)})")
                
                # Display uploaded images
                cols = st.columns(min(3, len(uploaded_files)))
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols[idx % 3]:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f"Image {idx+1}", use_container_width=True)
        
        # Register button
        if st.button("✨ Register User", type="primary", use_container_width=True):
            if not user_id:
                st.error("❌ Please enter a User ID")
            elif not uploaded_files:
                st.error("❌ Please upload at least one image")
            else:
                with st.spinner("Processing registration..."):
                    # Load images
                    images = []
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file)
                        images.append(image)
                    
                    # Prepare metadata
                    metadata = {}
                    if user_name:
                        metadata['name'] = user_name
                    if user_email:
                        metadata['email'] = user_email
                    if user_dept:
                        metadata['department'] = user_dept
                    
                    # Register user
                    success, message = auth_system.register_user(
                        user_id=user_id,
                        images=images,
                        metadata=metadata if metadata else None
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
    
    # Tab 2: Authentication (1:N)
    with tab2:
        st.header("User Authentication")
        st.markdown("Identify a user from the database")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Face Image")
            auth_file = st.file_uploader(
                "Choose a face image to authenticate",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="auth_upload"
            )
            
            if auth_file:
                auth_image = Image.open(auth_file)
                st.image(auth_image, caption="Image to Authenticate", use_container_width=True)
        
        with col2:
            st.subheader("Authentication Result")
            
            if auth_file and st.button("🔍 Authenticate", type="primary", use_container_width=True):
                with st.spinner("Authenticating..."):
                    auth_image = Image.open(auth_file)
                    
                    user_id, confidence, message = auth_system.authenticate_user(
                        image=auth_image,
                        return_confidence=True
                    )
                    
                    st.markdown("---")
                    
                    if user_id:
                        st.success(f"✅ **{message}**")
                        
                        # Show confidence meter
                        st.metric("Confidence Score", f"{confidence*100:.1f}%")
                        st.progress(confidence)
                        
                        # Show user info
                        user_info = auth_system.get_user_info(user_id)
                        if user_info:
                            st.markdown("### User Information")
                            st.write(f"**User ID:** {user_info['user_id']}")
                            if user_info.get('metadata'):
                                for key, value in user_info['metadata'].items():
                                    st.write(f"**{key.title()}:** {value}")
                    else:
                        st.error(f"❌ **{message}**")
                        
                        # Show confidence meter
                        st.metric("Best Match Score", f"{confidence*100:.1f}%")
                        st.progress(confidence)
                        
                        st.info(f"💡 Tip: Threshold is set to {threshold*100:.0f}%. Lower it in settings for more lenient matching.")
    
    # Tab 3: Verification (1:1)
    with tab3:
        st.header("User Verification")
        st.markdown("Verify if an image matches a specific user")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Get list of registered users
            users = auth_system.list_users()
            if users:
                user_ids = [user['user_id'] for user in users]
                verify_user_id = st.selectbox(
                    "Select User to Verify",
                    options=user_ids,
                    help="Choose the user to verify against"
                )
            else:
                st.warning("⚠️ No users registered yet. Please register users first.")
                verify_user_id = None
            
            st.subheader("Upload Face Image")
            verify_file = st.file_uploader(
                "Choose a face image to verify",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="verify_upload"
            )
            
            if verify_file:
                verify_image = Image.open(verify_file)
                st.image(verify_image, caption="Image to Verify", use_container_width=True)
        
        with col2:
            st.subheader("Verification Result")
            
            if verify_file and verify_user_id and st.button("✅ Verify", type="primary", use_container_width=True):
                with st.spinner("Verifying..."):
                    verify_image = Image.open(verify_file)
                    
                    verified, similarity, message = auth_system.verify_user(
                        user_id=verify_user_id,
                        image=verify_image
                    )
                    
                    st.markdown("---")
                    
                    if verified:
                        st.success(f"✅ **{message}**")
                        st.success(f"The image matches user '{verify_user_id}'")
                    else:
                        st.error(f"❌ **{message}**")
                        st.error(f"The image does NOT match user '{verify_user_id}'")
                    
                    # Show similarity meter
                    st.metric("Similarity Score", f"{similarity*100:.1f}%")
                    st.progress(similarity)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Face Authentication System | Powered by Deep Learning & Gabor Transform</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
