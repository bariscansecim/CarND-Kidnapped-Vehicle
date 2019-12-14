/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  //Normal distributions
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  //Generate particles with normal distribution with mean on GPS values.
  for(int i = 0; i < num_particles; i++) {
    Particle par;
    par.id = i;
    par.x = dist_x(gen);
    par.y = dist_y(gen);
    par.theta = dist_theta(gen);
    par.weight = 1.0;
    particles.push_back(par);
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  //Normal distributions for sensor noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) > 0.0001) {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (-cos(particles[i].theta + (yaw_rate * delta_t)) + cos(particles[i].theta));
      particles[i].theta += yaw_rate * delta_t;
    }else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (unsigned int i = 0 ; i < observations.size(); i++) {
    double minDist = numeric_limits<double>::max();
    int mapId = -1;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      double distX = observations[i].x - predicted[j].x;
      double distY = observations[i].y - predicted[j].y;
      double dist = distX * distX + distY * distY;
      if (minDist > dist) {
        minDist = dist;
        mapId = predicted[j].id;
      }
    }
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for (int i = 0; i < num_particles; i++) {
    vector <LandmarkObs>  predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      int landmarkId = map_landmarks.landmark_list[j].id_i;
      float landmarkX = map_landmarks.landmark_list[j].x_f;
      float landmarkY = map_landmarks.landmark_list[j].y_f;
      if (dist(particles[i].x , particles[i].y , landmarkX , landmarkY) < sensor_range) {
        predictions.push_back(LandmarkObs{ landmarkId,landmarkX, landmarkY});
      }
    }
    vector <LandmarkObs> transObs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double transX = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      double transY = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
      transObs.push_back(LandmarkObs{ observations[j].id, transX, transY});
    }
    dataAssociation(predictions, transObs);

    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < transObs.size(); j++) {
      int index = 0;
      while (transObs[j].id != predictions[index].id) {
        ++index;
      }
      particles[i].weight *= multi_gaussian(std_landmark[0], std_landmark[1], transObs[j].x, transObs[j].y, predictions[index].x, predictions[index].y);
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  
  vector<Particle> newParticles;
  
  for (int i = 0; i < num_particles; i++) {
    discrete_distribution<> dst2(weights.begin(), weights.end());
    newParticles.push_back(particles[dst2(gen)]);
  }
  particles = newParticles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}