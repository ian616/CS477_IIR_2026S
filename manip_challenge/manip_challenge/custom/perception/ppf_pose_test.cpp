#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <opencv2/core.hpp>
#include <opencv2/surface_matching/ppf_helpers.hpp>
#include <opencv2/surface_matching/ppf_match_3d.hpp>

namespace {

struct Options {
  std::string model_path;
  std::string scene_path;
  std::string output_path = "ppf_pose_result.yml";
  double model_scale = 1.0;
  double relative_sampling_step = 0.04;
  double relative_distance_step = 0.04;
  double relative_scene_sample_step = 1.0 / 5.0;
  double relative_scene_distance = 0.03;
  int num_angles = 30;
  int normal_neighbors = 20;
  int max_model_points = 12000;
  int max_scene_points = 5000;
};

void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --model <mesh.dae|obj|ply> --scene <scene.ply> [options]\n\n"
      << "Options:\n"
      << "  --model-scale <float>              Scale model vertices before matching. Default 1.0\n"
      << "  --output <result.yml>              Output pose file. Default ppf_pose_result.yml\n"
      << "  --relative-sampling-step <float>   PPF model sampling step. Default 0.04\n"
      << "  --relative-distance-step <float>   PPF distance bin step. Default 0.04\n"
      << "  --scene-sample-step <float>        Scene sample ratio. Default 0.2\n"
      << "  --scene-distance <float>           Scene distance threshold. Default 0.03\n"
      << "  --num-angles <int>                 Angle bins. Default 30\n"
      << "  --normal-neighbors <int>           Scene normal KNN. Default 20\n"
      << "  --max-model-points <int>           Vertex stride cap. Default 12000\n"
      << "  --max-scene-points <int>           Scene stride cap. Default 5000\n";
}

bool parse_args(int argc, char** argv, Options& opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const std::string& name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--model") opts.model_path = need_value(arg);
    else if (arg == "--scene") opts.scene_path = need_value(arg);
    else if (arg == "--output") opts.output_path = need_value(arg);
    else if (arg == "--model-scale") opts.model_scale = std::stod(need_value(arg));
    else if (arg == "--relative-sampling-step") opts.relative_sampling_step = std::stod(need_value(arg));
    else if (arg == "--relative-distance-step") opts.relative_distance_step = std::stod(need_value(arg));
    else if (arg == "--scene-sample-step") opts.relative_scene_sample_step = std::stod(need_value(arg));
    else if (arg == "--scene-distance") opts.relative_scene_distance = std::stod(need_value(arg));
    else if (arg == "--num-angles") opts.num_angles = std::stoi(need_value(arg));
    else if (arg == "--normal-neighbors") opts.normal_neighbors = std::stoi(need_value(arg));
    else if (arg == "--max-model-points") opts.max_model_points = std::stoi(need_value(arg));
    else if (arg == "--max-scene-points") opts.max_scene_points = std::stoi(need_value(arg));
    else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  return !opts.model_path.empty() && !opts.scene_path.empty();
}

cv::Mat stride_limit(const cv::Mat& input, int max_points) {
  if (max_points <= 0 || input.rows <= max_points) return input.clone();
  int stride = static_cast<int>(std::ceil(input.rows / static_cast<double>(max_points)));
  std::vector<int> keep;
  keep.reserve(max_points);
  for (int i = 0; i < input.rows; i += stride) keep.push_back(i);

  cv::Mat output(static_cast<int>(keep.size()), input.cols, input.type());
  for (int r = 0; r < static_cast<int>(keep.size()); ++r) {
    input.row(keep[r]).copyTo(output.row(r));
  }
  return output;
}

void print_bounds(const std::string& name, const cv::Mat& pc) {
  cv::Vec3f min_v(
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity());
  cv::Vec3f max_v(
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity());

  for (int i = 0; i < pc.rows; ++i) {
    const float* p = pc.ptr<float>(i);
    for (int k = 0; k < 3; ++k) {
      min_v[k] = std::min(min_v[k], p[k]);
      max_v[k] = std::max(max_v[k], p[k]);
    }
  }
  cv::Vec3f extent = max_v - min_v;
  std::cout << name << " points=" << pc.rows
            << " min=[" << min_v[0] << ", " << min_v[1] << ", " << min_v[2] << "]"
            << " max=[" << max_v[0] << ", " << max_v[1] << ", " << max_v[2] << "]"
            << " extent=[" << extent[0] << ", " << extent[1] << ", " << extent[2] << "]\n";
}

void append_mesh_points(const aiScene* scene, const aiMesh* mesh, double scale, std::vector<float>& rows) {
  for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
    const aiVector3D& v = mesh->mVertices[i];
    aiVector3D n(0.0f, 0.0f, 1.0f);
    if (mesh->HasNormals()) n = mesh->mNormals[i];
    float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len < 1e-6f) {
      n = aiVector3D(0.0f, 0.0f, 1.0f);
      len = 1.0f;
    }
    rows.push_back(static_cast<float>(v.x * scale));
    rows.push_back(static_cast<float>(v.y * scale));
    rows.push_back(static_cast<float>(v.z * scale));
    rows.push_back(n.x / len);
    rows.push_back(n.y / len);
    rows.push_back(n.z / len);
  }
}

cv::Mat load_model_with_normals(const std::string& path, double scale, int max_points) {
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(
      path,
      aiProcess_Triangulate |
      aiProcess_GenSmoothNormals |
      aiProcess_JoinIdenticalVertices |
      aiProcess_ImproveCacheLocality |
      aiProcess_PreTransformVertices);
  if (!scene || !scene->mRootNode) {
    throw std::runtime_error("Assimp failed to load model: " + std::string(importer.GetErrorString()));
  }

  std::vector<float> rows;
  for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
    append_mesh_points(scene, scene->mMeshes[i], scale, rows);
  }
  if (rows.empty()) throw std::runtime_error("Model has no vertices.");

  cv::Mat model(static_cast<int>(rows.size() / 6), 6, CV_32F, rows.data());
  cv::Mat copied = model.clone();
  return stride_limit(copied, max_points);
}

cv::Mat load_scene_with_normals(const std::string& path, int normal_neighbors, int max_points) {
  cv::Mat scene_xyz = cv::ppf_match_3d::loadPLYSimple(path.c_str(), 0);
  if (scene_xyz.empty()) throw std::runtime_error("Could not load scene PLY.");
  scene_xyz.convertTo(scene_xyz, CV_32F);
  scene_xyz = stride_limit(scene_xyz, max_points);

  cv::Mat scene_normals;
  int ok = cv::ppf_match_3d::computeNormalsPC3d(
      scene_xyz, scene_normals, normal_neighbors, true, cv::Vec3f(0.0f, 0.0f, 0.0f));
  if (ok == 0 || scene_normals.empty()) {
    throw std::runtime_error("Could not compute scene normals.");
  }
  scene_normals.convertTo(scene_normals, CV_32F);
  return scene_normals;
}

void write_result(const std::string& path, const std::vector<cv::ppf_match_3d::Pose3DPtr>& results) {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  fs << "num_results" << static_cast<int>(results.size());
  fs << "poses" << "[";
  for (const auto& pose : results) {
    fs << "{";
    fs << "votes" << static_cast<int>(pose->numVotes);
    fs << "residual" << pose->residual;
    fs << "translation" << "[" << pose->t[0] << pose->t[1] << pose->t[2] << "]";
    cv::Mat pose_mat(4, 4, CV_64F);
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) pose_mat.at<double>(r, c) = pose->pose(r, c);
    }
    fs << "pose_matrix" << pose_mat;
    fs << "}";
  }
  fs << "]";
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, opts)) {
    usage(argv[0]);
    return 2;
  }

  try {
    cv::Mat model = load_model_with_normals(opts.model_path, opts.model_scale, opts.max_model_points);
    cv::Mat scene = load_scene_with_normals(opts.scene_path, opts.normal_neighbors, opts.max_scene_points);
    print_bounds("model", model);
    print_bounds("scene", scene);

    cv::ppf_match_3d::PPF3DDetector detector(
        opts.relative_sampling_step, opts.relative_distance_step, opts.num_angles);
    std::cout << "training model...\n";
    detector.trainModel(model);

    std::vector<cv::ppf_match_3d::Pose3DPtr> results;
    std::cout << "matching scene...\n";
    detector.match(scene, results, opts.relative_scene_sample_step, opts.relative_scene_distance);
    std::cout << "num_results=" << results.size() << "\n";

    int limit = std::min<int>(results.size(), 5);
    for (int i = 0; i < limit; ++i) {
      const auto& p = results[i];
      std::cout << "result[" << i << "] votes=" << p->numVotes
                << " residual=" << p->residual
                << " t=[" << p->t[0] << ", " << p->t[1] << ", " << p->t[2] << "]\n";
      std::cout << cv::Mat(p->pose) << "\n";
    }
    write_result(opts.output_path, results);
    std::cout << "wrote " << opts.output_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
