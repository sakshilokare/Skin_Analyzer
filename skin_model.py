import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
from scipy import ndimage

class SkinAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = ['pores', 'redness', 'texture', 'spots', 'wrinkles', 'hydration', 'evenness']
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ML model"""
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Simulate training
        X_train = np.random.rand(500, len(self.feature_names))
        y_train = 70 + 20 * np.random.rand(500)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
    
    def analyze_skin(self, image, user_data):
        """Main analysis function"""
        features = self._extract_features(image, user_data)
        score = self._calculate_score(features, user_data)
        
        return {
            'score': score,
            'features': features,
            'conditions': self._diagnose_conditions(features, user_data),
            'recommendations': self._generate_recommendations(features, user_data, score),
            'risk_factors': self._identify_risk_factors(features, user_data)
        }
    
    def _extract_features(self, image, user_data):
        """Extract skin features"""
        if image:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
        else:
            # Use profile-based simulation
            return self._simulate_from_profile(user_data)
        
        # Simple feature extraction
        return {
            'pores': self._detect_pores_simple(gray),
            'redness': self._detect_redness_simple(img_array) if len(img_array.shape) == 3 else 4.0,
            'texture': self._analyze_texture_simple(gray),
            'spots': self._detect_spots_simple(gray),
            'wrinkles': self._detect_wrinkles_simple(gray),
            'hydration': self._estimate_hydration_simple(gray),
            'evenness': self._analyze_evenness_simple(gray)
        }
    
    def _simulate_from_profile(self, user_data):
        """Simulate features based on profile"""
        age = user_data['age']
        skin_type = user_data['skin_type']
        
        base_profiles = {
            'oily': {'pores': 7.2, 'redness': 4.1, 'hydration': 5.3},
            'dry': {'pores': 3.8, 'redness': 4.5, 'hydration': 3.2},
            'combination': {'pores': 5.8, 'redness': 4.0, 'hydration': 4.8},
            'normal': {'pores': 4.5, 'redness': 3.2, 'hydration': 6.1},
            'sensitive': {'pores': 4.2, 'redness': 6.8, 'hydration': 4.5}
        }
        
        base = base_profiles.get(skin_type, base_profiles['normal'])
        age_factor = max(0, (age - 20) * 0.02)
        
        return {
            'pores': base['pores'] + random.uniform(-0.5, 0.5),
            'redness': base['redness'] + random.uniform(-0.5, 0.5),
            'texture': 6.5 - age_factor + random.uniform(-0.5, 0.5),
            'spots': 2.5 + age_factor * 2 + random.uniform(-0.5, 0.5),
            'wrinkles': max(2.0, min(8.0, 2.0 + age_factor * 3)),
            'hydration': base['hydration'] + random.uniform(-0.5, 0.5),
            'evenness': 6.0 - age_factor + random.uniform(-0.5, 0.5)
        }
    
    def _detect_pores_simple(self, gray_image):
        edges = cv2.Canny(gray_image, 50, 150)
        pore_density = np.sum(edges) / (gray_image.shape[0] * gray_image.shape[1])
        return min(pore_density * 1000, 8)
    
    def _detect_redness_simple(self, rgb_image):
        red_channel = rgb_image[:, :, 0].astype(float)
        redness = np.mean(red_channel) / 255.0
        return min(redness * 12, 8)
    
    def _analyze_texture_simple(self, gray_image):
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        texture = np.var(laplacian)
        return min(texture / 1000, 7)
    
    def _detect_spots_simple(self, gray_image):
        _, dark_spots = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
        spot_density = np.sum(dark_spots) / (255 * dark_spots.size)
        return min(spot_density * 15, 7)
    
    def _detect_wrinkles_simple(self, gray_image):
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=5)
        line_density = len(lines) / (gray_image.size) * 10000 if lines is not None else 0
        return min(line_density / 2, 7)
    
    def _estimate_hydration_simple(self, gray_image):
        smoothness = 1 / (np.std(gray_image) + 1e-5)
        return min(smoothness * 0.001, 7)
    
    def _analyze_evenness_simple(self, gray_image):
        unevenness = ndimage.generic_filter(gray_image, np.std, size=5)
        overall_unevenness = np.mean(unevenness)
        return max(4, 8 - overall_unevenness / 10)
    
    def _calculate_score(self, features, user_data):
        weights = {
            'pores': 0.15, 'redness': 0.15, 'texture': 0.15,
            'spots': 0.15, 'wrinkles': 0.15, 'hydration': 0.15, 'evenness': 0.10
        }
        
        feature_scores = {
            'pores': max(0, 10 - features['pores']),
            'redness': max(0, 10 - features['redness']),
            'texture': features['texture'],
            'spots': max(0, 10 - features['spots']),
            'wrinkles': max(0, 10 - features['wrinkles']),
            'hydration': features['hydration'],
            'evenness': features['evenness']
        }
        
        total_score = sum(feature_scores[key] * weights[key] for key in weights) * 10
        age_factor = max(0.7, 1 - (user_data['age'] - 20) * 0.005)
        total_score *= age_factor
        
        return max(30, min(95, int(total_score)))
    
    def _diagnose_conditions(self, features, user_data):
        conditions = []
        if features['pores'] > 6:
            conditions.append({"name": "Enlarged Pores", "severity": "High" if features['pores'] > 7 else "Medium"})
        if features['redness'] > 5:
            conditions.append({"name": "Skin Redness", "severity": "High" if features['redness'] > 6 else "Medium"})
        if features['spots'] > 4:
            conditions.append({"name": "Hyperpigmentation", "severity": "High" if features['spots'] > 6 else "Medium"})
        if features['wrinkles'] > 4:
            conditions.append({"name": "Aging Signs", "severity": "High" if features['wrinkles'] > 6 else "Medium"})
        if features['hydration'] < 4:
            conditions.append({"name": "Dehydration", "severity": "High" if features['hydration'] < 3 else "Medium"})
        return conditions
    
    def _identify_risk_factors(self, features, user_data):
        risks = []
        if user_data['age'] > 35 and features['spots'] > 4:
            risks.append("Increased sun damage risk")
        if features['redness'] > 5 and user_data['skin_type'] == 'sensitive':
            risks.append("High sensitivity")
        return risks
    
    def _generate_recommendations(self, features, user_data, score):
        recommendations = []
        if features['hydration'] < 5:
            recommendations.extend(["Use hyaluronic acid", "Drink more water"])
        if features['pores'] > 5:
            recommendations.extend(["Use salicylic acid", "Weekly clay masks"])
        if features['redness'] > 5:
            recommendations.extend(["Use calming ingredients", "Avoid hot water"])
        
        recommendations.extend([
            "Cleanse twice daily",
            "Use sunscreen daily",
            "Get adequate sleep",
            "Eat healthy diet"
        ])
        return recommendations[:6]