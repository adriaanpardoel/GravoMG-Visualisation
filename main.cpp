#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <nfd.h>

#include "learnopengl/shader.h"

#include "gravomg/multigrid_solver.h"

#include <iostream>

#include "tinydir.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
typedef CGAL::Point_set_3<K::Point_3> PointSet;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void window_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
int screenWidth = 800;
int screenHeight = 600;

int frameBufferWidth;
int frameBufferHeight;

// camera
glm::vec3 cameraPosition;
glm::vec3 cameraFront;
glm::vec3 cameraUp;
glm::vec2 cameraRotation;

// mouse press callback
bool mousePressed;
bool mouseDragged;
glm::vec2 lastMousePosition;

// user settings
static float phi = 0.125f;
static float prevPhi = 0.0f;

int selectedPoint = 0;
int prevSelectedPoint = 0;

Eigen::RowVectorXd::Index coarsePoint;


glm::mat4 model, view, projection;
SurfaceMesh surface;
glm::vec3 rayCamPos;


std::vector<Eigen::MatrixXd> hierarchyVertices;
std::vector<std::vector<std::vector<int>>> hierarchyTriangles;
std::vector<Eigen::MatrixXi> hierarchyNeighbours;
std::vector<std::vector<int>> hierarchySampling;
std::vector<Eigen::SparseMatrix<double>> hierarchyProlongation;
std::vector<std::vector<bool>> hierarchyProlongationFallback;

std::vector<int> prolongationVertices;
std::vector<double> prolongationWeights;

int displayLevel = 0;


enum Prolongation {
    barycentricTriangle,
    barycentricEdge,
    fallback,
};


std::set<std::vector<int>> candidateEdges;
Prolongation prolongation;
std::vector<std::string> levelLabels;

std::vector<std::string> meshFiles;
std::vector<std::string> meshLabels;
int selectedMeshFile;
bool meshErrorPopup;

PointSet pointSet;


float maxDim;
float initialZoom;

glm::vec3 modelCenter;


// vertex buffer: [n fine vertices, m coarse vertices, 1 projected vertex] x 3d
// sampling buffer [n fine points] x 1 bool
// color buffer: [n fine points] x 3f
// edge buffer: [n fine edges, m coarse edges] x 2i
// prolongation edge buffer: [n candidate edges, 3 final triangle edges, 3 barycentric lines] x 2i
// triangle buffer: [n fine triangles, m coarse triangles] x 3i

unsigned int vertexArray, vertexBuffer, samplingBuffer, colorBuffer, edgeBuffer, prolongationEdgeBuffer, triangleBuffer;


static std::vector<std::vector<int>> generateTriangles(Eigen::MatrixXd& positions, Eigen::MatrixXi& neighbours) {
    std::vector<std::vector<int>> tris;
    tris.reserve(positions.rows() * neighbours.cols());
    for (int v1Idx = 0; v1Idx < positions.rows(); ++v1Idx) {
        int v2Idx, v3Idx;
        for (int col1 = 0; col1 < neighbours.cols(); col1++) {
            v2Idx = neighbours(v1Idx, col1);
            if (v2Idx < 0) break;
            // We iterate over the indices in order, so if the neighboring idx is lower than the current v1Idx,
            // it must have been considered before and be part of a triangle.
            if (v2Idx < v1Idx) continue;
            for (int col2 = col1 + 1; col2 < neighbours.cols(); col2++) {
                v3Idx = neighbours(v1Idx, col2);
                if (v3Idx < 0) break;
                if (v3Idx < v1Idx) continue;

                bool isNeighbour = false;
                for (int col3 = 0; col3 < neighbours.cols(); col3++) {
                    int v2Neighbour = neighbours(v2Idx, col3);
                    if (v2Neighbour < 0) break;
                    if (v2Neighbour == v3Idx) {
                        isNeighbour = true;
                        break;
                    }
                }

                if (isNeighbour) {
                    tris.push_back({v1Idx, v2Idx, v3Idx});
                }
            }
        }
    }
    tris.shrink_to_fit();

    return tris;
}

void constructHierarchy() {
    auto filePath = meshFiles[selectedMeshFile];
    auto ext = filePath.substr(filePath.find_last_of('.') + 1);

    int nPoints = (ext == "xyz" ? pointSet.size() : surface.num_vertices());
    Eigen::MatrixXd positions(nPoints, 3);
    Eigen::MatrixXi neighbours(nPoints, nPoints);
    Eigen::SparseMatrix<double> mass(nPoints, nPoints);

    neighbours.setConstant(-1);
    int maxNeighbours = 0;

    if (ext == "xyz") {
        // point cloud
        auto averageSpacing = CGAL::compute_average_spacing<CGAL::Parallel_if_available_tag>(pointSet, 10, CGAL::parameters::all_default());
        float r2 = averageSpacing * averageSpacing;

        int v = 0;
        for (auto p : pointSet.points()) {
            positions(v, 0) = p.x();
            positions(v, 1) = p.y();
            positions(v, 2) = p.z();

            int w = 0;
            int col = 0;
            for (auto q : pointSet.points()) {
                float d2 = squared_distance(p, q);
                if (p != q && d2 <= r2) {
                    neighbours(v, col++) = w;
                }
                w++;
            }

            maxNeighbours = std::max(maxNeighbours, col);
            v++;
        }
    } else {
        // surface mesh
        try {
            CGAL::Polygon_mesh_processing::triangulate_faces(surface);

            for (auto v : surface.vertices()) {
                auto p = surface.point(v);

                positions(v.idx(), 0) = p.x();
                positions(v.idx(), 1) = p.y();
                positions(v.idx(), 2) = p.z();

                int col = 0;
                for (auto outEdge : CGAL::halfedges_around_source(v, surface)) {
                    auto neighbour = surface.target(outEdge);
                    neighbours(v.idx(), col++) = neighbour.idx();
                }

                maxNeighbours = std::max(maxNeighbours, col);
            }
        } catch (...) {
            meshErrorPopup = true;
        }
    }

    neighbours.conservativeResize(neighbours.rows(), maxNeighbours);

    mass.setIdentity();

    GravoMG::MultigridSolver solver(positions, neighbours, mass);
    solver.debug = true;
    solver.lowBound = 3;
    solver.ratio = 1.0f / phi;
    solver.buildHierarchy();

    hierarchyVertices = solver.levelV;
    hierarchyTriangles = solver.allTriangles;
    hierarchyNeighbours = solver.neighHierarchy;
    hierarchySampling = solver.samples;
    hierarchyProlongation = solver.U;
    hierarchyProlongationFallback = solver.prolongationFallback;

    hierarchyVertices.insert(hierarchyVertices.begin(), positions);
    hierarchyTriangles.insert(hierarchyTriangles.begin(), generateTriangles(positions, neighbours));
    hierarchyNeighbours.insert(hierarchyNeighbours.begin(), neighbours);

    if (hierarchyProlongation.empty()) {
        meshErrorPopup = true;
    }

    Eigen::RowVectorXd minVal = positions.colwise().minCoeff();
    Eigen::RowVectorXd maxVal = positions.colwise().maxCoeff();

    modelCenter = { (minVal.x() + maxVal.x()) / 2.0f, (minVal.y() + maxVal.y()) / 2.0f, (minVal.z() + maxVal.z()) / 2.0f };
    maxDim = max(max(maxVal.x() - minVal.x(), maxVal.y() - minVal.y()), maxVal.z() - minVal.z());

    cameraPosition = { 0, 0, maxVal.z() - modelCenter.z + 2 * maxDim };
    cameraFront = { 0, 0, -1 };
    cameraUp = { 0, 1, 0 };
    cameraRotation = { 0, 0 };
    initialZoom = cameraPosition.z;
}

static int countEdges(Eigen::MatrixXi &neighbours) {
    return ((neighbours.array() >= 0).cast<int>().sum()) / 2;
}

static std::vector<std::vector<int>> getEdges(Eigen::MatrixXi &neighbours) {
    std::vector<std::vector<int>> res;

    for (int i = 0; i < neighbours.rows(); i++) {
        for (int j = 0; j < neighbours.cols(); j++) {
            int to = neighbours(i, j);

            if (to < 0) break;
            if (i >= to) continue;

            res.push_back({ i, to });
        }
    }

    return res;
}

static void bufferHierarchyData(Eigen::MatrixXd &fineVertexPositions, Eigen::MatrixXd &coarseVertexPositions,
                         std::vector<int> &sampling, Eigen::VectorXi &neighbourhoods, std::vector<Eigen::Vector3f> &neighbourhoodColors,
                         Eigen::MatrixXi &fineNeighbours, Eigen::MatrixXi &coarseNeighbours,
                         std::vector<std::vector<int>> &fineTriangles, std::vector<std::vector<int>> &coarseTriangles) {
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, (void*)0);
    int vertexBufferSize = (fineVertexPositions.rows() + coarseVertexPositions.rows() + 1) * 3 * sizeof(GLdouble);
    glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_STATIC_DRAW);

    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> rowMajor = fineVertexPositions;
    int fineVertexDataSize = fineVertexPositions.rows() * 3 * sizeof(GLdouble);
    glBufferSubData(GL_ARRAY_BUFFER, 0, fineVertexDataSize, rowMajor.data());

    rowMajor = coarseVertexPositions;
    int coarseVertexDataSize = coarseVertexPositions.rows() * 3 * sizeof(GLdouble);
    glBufferSubData(GL_ARRAY_BUFFER, fineVertexDataSize, coarseVertexDataSize, rowMajor.data());

    glBindBuffer(GL_ARRAY_BUFFER, samplingBuffer);
    glVertexAttribIPointer(1, 1, GL_BYTE, 0, (void*)0);
    auto* samplingData = new GLboolean[fineVertexPositions.rows()]();
    for (int v : sampling) {
        samplingData[v] = 1;
    }
    glBufferData(GL_ARRAY_BUFFER, fineVertexPositions.rows() * sizeof(GLboolean), samplingData, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    std::vector<Eigen::Vector4f> colorData(fineVertexPositions.rows());
    for (int v = 0; v < fineVertexPositions.rows(); v++) {
        auto col = neighbourhoodColors[neighbourhoods[v]];
        colorData[v] = Eigen::Vector4f(col.x(), col.y(), col.z(), 1);
    }
    glBufferData(GL_ARRAY_BUFFER, fineVertexPositions.rows() * sizeof(Eigen::Vector4f), &colorData[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, edgeBuffer);
    auto edges = getEdges(fineNeighbours);
    auto coarseEdges = getEdges(coarseNeighbours);
    for (auto& e : coarseEdges) {
        e[0] += fineVertexPositions.rows();
        e[1] += fineVertexPositions.rows();
    }
    edges.insert(edges.end(), coarseEdges.begin(), coarseEdges.end());
    std::vector<int> flattenedEdges;
    for (auto e : edges) {
        flattenedEdges.push_back(e[0]);
        flattenedEdges.push_back(e[1]);
    }
    glBufferData(GL_ARRAY_BUFFER, edges.size() * 2 * sizeof(GLuint), &flattenedEdges[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, triangleBuffer);
    auto triangles = fineTriangles;
    triangles.insert(triangles.end(), coarseTriangles.begin(), coarseTriangles.end());
    for (int i = fineTriangles.size(); i < triangles.size(); i++) {
        triangles[i][0] += fineVertexPositions.rows();
        triangles[i][1] += fineVertexPositions.rows();
        triangles[i][2] += fineVertexPositions.rows();
    }
    std::vector<int> flattenedTriangles;
    for (auto t : triangles) {
        flattenedTriangles.push_back(t[0]);
        flattenedTriangles.push_back(t[1]);
        flattenedTriangles.push_back(t[2]);
    }
    glBufferData(GL_ARRAY_BUFFER, triangles.size() * 3 * sizeof(GLuint), &flattenedTriangles[0], GL_STATIC_DRAW);
}

static void bufferProlongationData(Prolongation prolongation, Eigen::MatrixXd &fineVertexPositions, Eigen::MatrixXd &coarseVertexPositions,
                                   Eigen::Vector3d &projectedVertexPosition, int finePoint, std::set<std::vector<int>> &candidateEdges, std::vector<int> &prolongationVertices) {
    glBindVertexArray(vertexArray);

    if (prolongation != fallback) {
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        int projectedVertexOffset = (fineVertexPositions.rows() + coarseVertexPositions.rows()) * 3 * sizeof(GLdouble);
        glBufferSubData(GL_ARRAY_BUFFER, projectedVertexOffset, 3 * sizeof(GLdouble), projectedVertexPosition.data());
    }

    glBindBuffer(GL_ARRAY_BUFFER, prolongationEdgeBuffer);

    std::vector<std::vector<int>> prolongationEdges(candidateEdges.begin(), candidateEdges.end());

    int nFinePoints = fineVertexPositions.rows();
    for (auto& e : prolongationEdges) {
        e[0] += nFinePoints;
        e[1] += nFinePoints;
    }

    int projectedVertex = fineVertexPositions.rows() + coarseVertexPositions.rows();

    switch (prolongation) {
        case barycentricTriangle:
            for (int i = 0; i < 3; i++) {
                prolongationEdges.push_back({ prolongationVertices[i] + nFinePoints, prolongationVertices[(i + 1) % 3] + nFinePoints }); // final triangle
            }
            for (int i = 0; i < 3; i++) {
                prolongationEdges.push_back({ projectedVertex, prolongationVertices[i] + nFinePoints }); // barycentric lines
            }
            break;
        case barycentricEdge:
            prolongationEdges.push_back({ prolongationVertices[0] + nFinePoints, prolongationVertices[1] + nFinePoints });
            break;
        case fallback:
            for (int i = 0; i < 3; i++) {
                prolongationEdges.push_back({ finePoint, prolongationVertices[i] + nFinePoints }); // lines to closest points
            }
            break;
    }

    std::vector<int> flattenedEdges;
    for (auto e : prolongationEdges) {
        flattenedEdges.push_back(e[0]);
        flattenedEdges.push_back(e[1]);
    }

    glBufferData(GL_ARRAY_BUFFER, prolongationEdges.size() * 2 * sizeof(GLuint), &flattenedEdges[0], GL_STATIC_DRAW);
}

bool showProlongation(Prolongation prolongation) {
    std::vector<int> candidates;

    if (prolongation == fallback) {
        for (int v = 0; v < hierarchyProlongationFallback[displayLevel].size(); v++) {
            if (hierarchyProlongationFallback[displayLevel][v]) {
                candidates.push_back(v);
            }
        }
    } else {
        for (int v = 0; v < hierarchyProlongation[displayLevel].rows(); v++) {
            Eigen::RowVectorXd weightsRow = hierarchyProlongation[displayLevel].row(v);
            int nElements = (weightsRow.array() > 0.0).cast<int>().sum();
            if (!hierarchyProlongationFallback[displayLevel][v] &&
                    ((prolongation == barycentricTriangle && nElements == 3) ||
                    (prolongation == barycentricEdge && nElements < 3))) {
                candidates.push_back(v);
            }
        }
    }

    if (candidates.empty()) {
        return false;
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, candidates.size() - 1);
    selectedPoint = candidates.at(dis(gen));

    return true;
}

static Eigen::Vector3f rainbowColormap(double ratio)
{
    //we want to normalize ratio so that it fits in to 6 regions
    //where each region is 256 units long
    int normalized = int(ratio * 256 * 6);

    //find the region for this position
    int region = normalized / 256;

    //find the distance to the start of the closest region
    int x = normalized % 256;

    int r = 0, g = 0, b = 0;
    switch (region)
    {
        case 0: r = 255; g = 0;   b = 0;   g += x; break;
        case 1: r = 255; g = 255; b = 0;   r -= x; break;
        case 2: r = 0;   g = 255; b = 0;   b += x; break;
        case 3: r = 0;   g = 255; b = 255; g -= x; break;
        case 4: r = 0;   g = 0;   b = 255; r += x; break;
        case 5: r = 255; g = 0;   b = 255; b -= x; break;
    }

    return Eigen::Vector3f(r, g, b) / 255.0f;
}

void addMeshFile(const std::string& filePath) {
    auto i = std::distance(meshFiles.begin(), std::find(meshFiles.begin(), meshFiles.end(), filePath));
    if (i < meshFiles.size()) {
        selectedMeshFile = i;
        return;
    }

    std::string fileName = filePath.substr(filePath.find_last_of("/\\") + 1);

    std::string fileLabel = fileName;
    int labelIndex = 1;
    while (std::distance(meshLabels.begin(), std::find(meshLabels.begin(), meshLabels.end(), fileLabel)) < meshLabels.size()) {
        fileLabel = fileName + " (" + std::to_string(labelIndex++) + ")";
    }

    meshFiles.push_back(filePath);
    meshLabels.push_back(fileLabel);
    selectedMeshFile = meshFiles.size() - 1;
}

void updateBuffers(bool updateHierarchyBuffers, bool updateProlongationBuffers) {
    if (updateHierarchyBuffers) {
        selectedPoint = 0;
    }

    std::vector<Eigen::Vector3f> neighbourhoodColors(hierarchyVertices[displayLevel + 1].rows());
    for (int neighbourhood = 0; neighbourhood < hierarchyVertices[displayLevel + 1].rows(); neighbourhood++) {
        neighbourhoodColors[neighbourhood] = rainbowColormap((float)neighbourhood / hierarchyVertices[displayLevel + 1].rows());
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(neighbourhoodColors.begin(), neighbourhoodColors.end(), g);

    Eigen::VectorXi neighbourhoods(hierarchyProlongation[displayLevel].rows());
    for (int v = 0; v < hierarchyProlongation[displayLevel].rows(); v++) {
        Eigen::RowVectorXd weightsRow = hierarchyProlongation[displayLevel].row(v);
        Eigen::RowVectorXd::Index maxIndex;
        weightsRow.maxCoeff(&maxIndex);
        neighbourhoods[v] = maxIndex;
    }

    coarsePoint = neighbourhoods[selectedPoint];

    Eigen::RowVectorXd weightsRow = hierarchyProlongation[displayLevel].row(selectedPoint);

    int nElements = (weightsRow.array() > 0.0).cast<int>().sum();

    if (hierarchyProlongationFallback[displayLevel][selectedPoint]) {
        prolongation = fallback;
    } else if (nElements == 3) {
        prolongation = barycentricTriangle;
    } else {
        prolongation = barycentricEdge;
    }

    prolongationVertices.clear();
    prolongationWeights.clear();

    for (int j = 0; j < hierarchyProlongation[displayLevel].cols(); j++) {
        if (weightsRow[j] > 0) {
            prolongationVertices.push_back(j);
            prolongationWeights.push_back(weightsRow[j]);
        }
    }

    if (nElements == 1) {
        prolongationVertices[1] = prolongationVertices[0];
        prolongationWeights[1] = 0;
    }

    auto contains = [](auto v, auto x) { return std::find(v.begin(), v.end(), x) != v.end(); };

    candidateEdges.clear();
    for (auto t : hierarchyTriangles[displayLevel + 1]) {
        if (!contains(t, coarsePoint)) continue;

        candidateEdges.insert({ std::min(t[0], t[1]), std::max(t[0], t[1]) });
        candidateEdges.insert({ std::min(t[1], t[2]), std::max(t[1], t[2]) });
        candidateEdges.insert({ std::min(t[2], t[0]), std::max(t[2], t[0]) });
    }

    Eigen::Vector3d projectedVertexPosition = weightsRow * hierarchyVertices[displayLevel + 1];

    levelLabels.clear();
    for (int l = 0; l < hierarchyVertices.size() - 1; l++) {
        std::ostringstream stringStream;
        stringStream << l << " to " << (l + 1);
        levelLabels.push_back(stringStream.str());
    }

    if (updateHierarchyBuffers)
        bufferHierarchyData(hierarchyVertices[displayLevel], hierarchyVertices[displayLevel + 1], hierarchySampling[displayLevel], neighbourhoods, neighbourhoodColors, hierarchyNeighbours[displayLevel], hierarchyNeighbours[displayLevel + 1], hierarchyTriangles[displayLevel], hierarchyTriangles[displayLevel + 1]);

    if (updateProlongationBuffers)
        bufferProlongationData(prolongation, hierarchyVertices[displayLevel], hierarchyVertices[displayLevel + 1], projectedVertexPosition, selectedPoint, candidateEdges, prolongationVertices);
}

void loadMeshFromFile(const std::string& filePath) {
    auto ext = filePath.substr(filePath.find_last_of('.') + 1);

    surface.clear();
    pointSet.clear();

    std::ifstream in(filePath);

    if (ext == "off") {
        meshErrorPopup = !CGAL::IO::read_OFF(in, surface);
    } else if (ext == "obj") {
        meshErrorPopup = !CGAL::IO::read_OBJ(in, surface);
    } else if (ext == "stl") {
        meshErrorPopup = !CGAL::IO::read_STL(in, surface);
    } else if (ext == "ply") {
        meshErrorPopup = !CGAL::IO::read_PLY(in, surface);
    } else if (ext == "ts") {
        meshErrorPopup = !CGAL::IO::read_GOCAD(in, surface);
    } else if (ext == "xyz") {
        meshErrorPopup = !CGAL::IO::read_XYZ(in, pointSet);
    }
}

void drawSamplingEdgesAndVertices(Shader &defaultShader, Shader &samplingShader, int nFineEdges) {
    // draw edges
    defaultShader.use();
    defaultShader.setBool("useUniformColor", false);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
    glDrawElements(GL_LINES, 2 * nFineEdges, GL_UNSIGNED_INT, 0);

    // draw vertices
    samplingShader.use();
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glDrawArrays(GL_POINTS, 0, hierarchyVertices[displayLevel].rows());
    glDisableVertexAttribArray(1);
}

void drawNeighbourhoodsEdgesAndVertices(Shader &defaultShader, int nFineEdges) {
    // draw edges
    defaultShader.use();
    defaultShader.setBool("useUniformColor", false);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
    glDrawElements(GL_LINES, 2 * nFineEdges, GL_UNSIGNED_INT, 0);

    // draw vertices
    glEnableVertexAttribArray(2);
    glDrawArrays(GL_POINTS, 0, hierarchyVertices[displayLevel].rows());
    glDisableVertexAttribArray(2);
}

void drawProlongationEdgesAndVertices(Shader &defaultShader, int nFineEdges, int nCoarseEdges) {
    // draw edges
    defaultShader.use();
    defaultShader.setBool("useUniformColor", false);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
    glDrawElements(GL_LINES, 2 * nCoarseEdges, GL_UNSIGNED_INT, (void*) (nFineEdges * 2 * sizeof(GLuint)));

    // draw candidate edges
    defaultShader.setBool("useUniformColor", true);
    defaultShader.setVec4("uColor", glm::vec4(1, 1, 0, 1));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prolongationEdgeBuffer);
    glDrawElements(GL_LINES, 2 * candidateEdges.size(), GL_UNSIGNED_INT, 0);

    switch (prolongation) {
        case barycentricTriangle:
            // draw final triangle
            defaultShader.setVec4("uColor", glm::vec4(0, 0.8f, 1, 1));
            glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, (void*) (candidateEdges.size() * 2 * sizeof(GLuint)));

            // draw barycentric lines
            defaultShader.setVec4("uColor", glm::vec4(1, 0, 0, 1));
            glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, (void*) ((candidateEdges.size() + 3) * 2 * sizeof(GLuint)));
            break;
        case barycentricEdge:
            // draw barycentric edge
            defaultShader.setVec4("uColor", glm::vec4(1, 0, 0, 1));
            glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, (void*) (candidateEdges.size() * 2 * sizeof(GLuint)));
            break;
        case fallback:
            // draw lines to closest points
            defaultShader.setVec4("uColor", glm::vec4(1, 0, 0, 1));
            glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, (void*) (candidateEdges.size() * 2 * sizeof(GLuint)));
            break;
    }

    if (prolongation != fallback) {
        // draw coarse point
        defaultShader.setVec4("uColor", glm::vec4(1, 1, 0, 1));
        glDrawArrays(GL_POINTS, hierarchyVertices[displayLevel].rows() + coarsePoint, 1);

        // draw projected point
        defaultShader.setVec4("uColor", glm::vec4(0.6f, 0, 0.8f, 1));
        glDrawArrays(GL_POINTS, hierarchyVertices[displayLevel].rows() + hierarchyVertices[displayLevel + 1].rows(), 1);
    }

    // draw selected point
    defaultShader.setVec4("uColor", glm::vec4(1, 0, 1, 1));
    glDrawArrays(GL_POINTS, selectedPoint, 1);
}

void drawTriangles(Shader &defaultShader, bool fine) {
    defaultShader.use();
    defaultShader.setBool("useUniformColor", true);
    defaultShader.setVec4("uColor", glm::vec4(0.6));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangleBuffer);

    if (fine) {
        glDrawElements(GL_TRIANGLES, 3 * hierarchyTriangles[displayLevel].size(), GL_UNSIGNED_INT, 0);
    } else {
        glDrawElements(GL_TRIANGLES, 3 * hierarchyTriangles[displayLevel + 1].size(), GL_UNSIGNED_INT, (void*) (hierarchyTriangles[displayLevel].size() * 3 * sizeof(GLuint)));
    }
}

void drawModel(const std::function<void()>& drawEdgesAndVertices, const std::function<void()>& drawTriangles) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    drawEdgesAndVertices();
    drawTriangles();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    // fill depth buffer
    drawTriangles();

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthFunc(GL_LEQUAL);

    drawEdgesAndVertices();

    glDisable(GL_DEPTH_TEST);
}

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "GravoMG Visualisation", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwGetWindowSize(window, &screenWidth, &screenHeight);
    glfwGetFramebufferSize(window, &frameBufferWidth, &frameBufferHeight);

    // glfw callbacks
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);

    // initialize glew
    // ---------------------------------------
    GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    NFD_Init();

    // Retrieve mesh files
    tinydir_dir dir;
    tinydir_open_sorted(&dir, "meshes");

    for (int i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        tinydir_readfile_n(&dir, &file, i);

        if (file.is_reg) {
            meshFiles.push_back(file.path);
            meshLabels.push_back(file.name);
        }
    }

    tinydir_close(&dir);

    auto cactusIndex = std::distance(meshLabels.begin(), std::find(meshLabels.begin(), meshLabels.end(), "cactus.off"));
    selectedMeshFile = std::max(0, (int) cactusIndex);

    // build and compile our shader program
    // ------------------------------------
    Shader defaultShader("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");
    Shader samplingShader("shaders/vertex_shader_sampling.glsl", "shaders/fragment_shader.glsl");

    samplingShader.setVec3("samplingColor", 1, 1, 0);
    samplingShader.setVec3("selectionColor", 1, 0, 1);

    loadMeshFromFile(meshFiles[selectedMeshFile]);

    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &samplingBuffer);
    glGenBuffers(1, &colorBuffer);
    glGenBuffers(1, &edgeBuffer);
    glGenBuffers(1, &prolongationEdgeBuffer);
    glGenBuffers(1, &triangleBuffer);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // render imgui window
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Options", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration);
        ImGui::SetWindowPos(ImVec2(10, 10));
        ImGui::SetWindowSize(ImVec2(300, 120));

        ImGui::DragFloat("phi", &phi, 0.005f);
        if (!ImGui::IsItemActive() && phi != prevPhi) {
            prevPhi = phi;

            constructHierarchy();

            displayLevel = 0;
            updateBuffers(true, true);
        }

        if (selectedPoint != prevSelectedPoint) {
            prevSelectedPoint = selectedPoint;
            updateBuffers(false, true);
        }

        if (ImGui::BeginCombo("Level", levelLabels[displayLevel].c_str(), 0))
        {
            for (int l = 0; l < hierarchyVertices.size() - 1; l++)
            {
                const bool is_selected = (displayLevel == l);
                if (ImGui::Selectable(levelLabels[l].c_str(), is_selected)) {
                    displayLevel = l;
                    updateBuffers(true, true);
                }

                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (ImGui::BeginCombo("Mesh", meshLabels[selectedMeshFile].c_str(), 0))
        {
            for (int i = 0; i < meshFiles.size(); i++)
            {
                const bool is_selected = (selectedMeshFile == i);
                if (ImGui::Selectable(meshLabels[i].c_str(), is_selected)) {
                    selectedMeshFile = i;
                    loadMeshFromFile(meshFiles[selectedMeshFile]);
                    constructHierarchy();
                    displayLevel = 0;
                    updateBuffers(true, true);
                }

                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }

            if (ImGui::Selectable("Select file...")) {
                nfdchar_t *outPath;
                nfdfilteritem_t filterItem[1] = { { "Object files", "obj,off" } };
                nfdresult_t result = NFD_OpenDialog(&outPath, filterItem, 1, NULL);

                if (result == NFD_OKAY) {
                    addMeshFile(std::string(outPath));
                    loadMeshFromFile(meshFiles[selectedMeshFile]);
                    constructHierarchy();
                    displayLevel = 0;
                    updateBuffers(true, true);
                }
            }
            ImGui::EndCombo();
        }

        ImGui::End();

        ImGui::Begin("Prolongation", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration);
        ImGui::SetWindowPos(ImVec2(screenWidth - 300 - 10, 10));
        ImGui::SetWindowSize(ImVec2(300, 120));

        const char* prolongationDesc;
        switch (prolongation) {
            case barycentricTriangle:
                prolongationDesc = "Barycentric coordinates in triangle";
                break;
            case barycentricEdge:
                prolongationDesc = "Barycentric coordinates on edge";
                break;
            case fallback:
                prolongationDesc = "Inverse distance weights";
                break;
        }

        ImGui::Text("Prolongation:");
        ImGui::Text("%s", prolongationDesc);
        ImGui::Dummy(ImVec2(0.0f, 8.0f));
        ImGui::Text("v1: %f", prolongationWeights[0]);
        ImGui::Text("v2: %f", prolongationWeights[1]);

        if (prolongation != barycentricEdge) {
            ImGui::Text("v3: %f", prolongationWeights[2]);
        }

        ImGui::End();

        ImGui::Begin("Show prolongation", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration);
        ImGui::SetWindowPos(ImVec2(screenWidth - 620, 10));
        ImGui::SetWindowSize(ImVec2(300, 120));

        bool prolongationFound = true;

        if (ImGui::Button("Show barycentric triangle")) {
            prolongationFound = showProlongation(barycentricTriangle);
        }
        if (ImGui::Button("Show barycentric edge")) {
            prolongationFound = showProlongation(barycentricEdge);
        }
        if (ImGui::Button("Show fallback")) {
            prolongationFound = showProlongation(fallback);
        }

        if (!prolongationFound) {
            ImGui::OpenPopup("Not found");
        }

        if (meshErrorPopup) {
            ImGui::OpenPopup("File error");
            meshErrorPopup = false;
        }

        // Always center this window when appearing
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();

        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("File error", NULL, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::Text("Unfortunately, the file could not be opened :(");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("OK", ImVec2(-1.0f, 0.0f))) { ImGui::CloseCurrentPopup(); }
            ImGui::SetItemDefaultFocus();
            ImGui::EndPopup();
        }

        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("Not found", NULL, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::Text("Unfortunately, a prolongation of that\ntype could not be found :(");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("OK", ImVec2(-1.0f, 0.0f))) { ImGui::CloseCurrentPopup(); }
            ImGui::SetItemDefaultFocus();
            ImGui::EndPopup();
        }

        ImGui::End();

        // input
        // -----
        processInput(window);

        int vpWidth = 0.333 * frameBufferWidth;
        int vpHeight = 0.9 * frameBufferHeight;

        // pass projection matrix to shader (note that in this case it could change every frame)
        projection = glm::perspective(glm::radians(45.0f), (float)vpWidth / (float)vpHeight, 0.1f, maxDim * 100.0f);
        defaultShader.setMat4("projection", projection);
        samplingShader.setMat4("projection", projection);

        // camera/view transformation
        view = glm::lookAt(cameraPosition, cameraPosition + cameraFront, cameraUp);
        view = glm::rotate(view, cameraRotation.x, glm::vec3(1, 0, 0));
        view = glm::rotate(view, cameraRotation.y, glm::vec3(0, 1, 0));
        defaultShader.setMat4("view", view);
        samplingShader.setMat4("view", view);

        // calculate the model matrix and pass it to shader before drawing
        model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        model = glm::translate(model, -modelCenter);
        defaultShader.setMat4("model", model);
        samplingShader.setMat4("model", model);

        samplingShader.setInt("selectedVertex", selectedPoint);

        // set point size
        glPointSize(5 * initialZoom / glm::length(cameraPosition));

        // enable smooth lines
        glEnable(GL_LINE_SMOOTH);

        // render
        // ------
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!hierarchyProlongation.empty()) {
            int nFineEdges = countEdges(hierarchyNeighbours[displayLevel]);
            int nCoarseEdges = countEdges(hierarchyNeighbours[displayLevel + 1]);

            glBindVertexArray(vertexArray);
            glEnableVertexAttribArray(0);

            // sampling (left pane)
            glViewport(0, 0, vpWidth, vpHeight);

            drawModel([&]() { drawSamplingEdgesAndVertices(defaultShader, samplingShader, nFineEdges); },
                      [&]() { drawTriangles(defaultShader, true); });

            // neighbourhoods (middle pane)
            glViewport(vpWidth, 0, vpWidth, vpHeight);

            drawModel([&]() { drawNeighbourhoodsEdgesAndVertices(defaultShader, nFineEdges); },
                      [&]() { drawTriangles(defaultShader, true); });

            // prolongation (right pane)
            glViewport(2 * vpWidth, 0, vpWidth, vpHeight);

            drawModel([&]() { drawProlongationEdgesAndVertices(defaultShader, nFineEdges, nCoarseEdges); },
                      [&]() { drawTriangles(defaultShader, false); });
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &vertexArray);
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &samplingBuffer);
    glDeleteBuffers(1, &colorBuffer);
    glDeleteBuffers(1, &edgeBuffer);
    glDeleteBuffers(1, &prolongationEdgeBuffer);
    glDeleteBuffers(1, &triangleBuffer);

    NFD_Quit();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    frameBufferWidth = width;
    frameBufferHeight = height;
}

// glfw: whenever a mouse button is pressed, this callback is  called
// ------------------------------------------------------------------
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && !hierarchyProlongation.empty()) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            mouseDragged = false;
        } else {
            mousePressed = false;

            if (!mouseDragged) {
                int vpWidth = 0.333 * screenWidth;
                int vpHeight = 0.9 * screenHeight;

                if (lastMousePosition.x < vpWidth && lastMousePosition.y > 0.1 * screenHeight) {
                    float x = (2.0f * ((int)lastMousePosition.x % vpWidth)) / vpWidth - 1.0f;
                    float y = 1.0f - (2.0f * (lastMousePosition.y - 0.1f * screenHeight)) / vpHeight;
                    float z = 1.0f;
                    glm::vec3 ray_nds(x, y, z);

                    glm::vec4 ray_clip(ray_nds.x, ray_nds.y, -1.0, 1.0);

                    glm::vec4 ray_eye = inverse(projection) * ray_clip;

                    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);

                    glm::vec3 ray_wor = inverse(model) * inverse(view) * ray_eye;
                    ray_wor = normalize(ray_wor);

                    auto rot = glm::mat4(1.0f);
                    rot = glm::rotate(rot, cameraRotation.x, glm::vec3(1, 0, 0));
                    rot = glm::rotate(rot, cameraRotation.y, glm::vec3(0, 1, 0));

                    rayCamPos = inverse(model) * inverse(rot) * glm::vec4(cameraPosition, 1.0f);

                    float tMin = INFINITY;
                    int vMin;
                    bool intersection = false;
                    int intersections = 0;

                    float radius2 = pow(maxDim, 1.5) * 0.0001f; // heuristic that works well

                    for (int v = 0; v < hierarchyVertices[displayLevel].rows(); v++) {
                        float t0, t1; // solutions for t if the ray intersects

                        // geometric solution
                        auto row = hierarchyVertices[displayLevel].row(v);
                        glm::vec3 vPos = { row.x(), row.y(), row.z() };
                        glm::vec3 L = vPos - rayCamPos;
                        float tca = glm::dot(L, ray_wor);
                        // if (tca < 0) return false;
                        float d2 = glm::dot(L, L) - tca * tca;
                        if (d2 > min(1.0f, tca / maxDim) * radius2) continue; // heuristic that works well
                        float thc = sqrt(radius2 - d2);
                        t0 = tca - thc;
                        t1 = tca + thc;

                        if (t0 > t1) std::swap(t0, t1);

                        if (t0 < 0) {
                            t0 = t1; // if t0 is negative, let's use t1 instead
                            if (t0 < 0) continue; // both t0 and t1 are negative
                        }

                        float t = t0;

                        if (t < tMin) {
                            tMin = t;
                            vMin = v;
                            intersection = true;
                            intersections++;
                        }
                    }

                    if (intersection) {
                        selectedPoint = vMin;
                    }
                }
            }

            mouseDragged = false;
        }
    }
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    if (mousePressed) {
        mouseDragged = true;

        float xRot = (ypos - lastMousePosition.y) * 2 * M_PI / screenHeight;
        float yRot = (xpos - lastMousePosition.x) * 2 * M_PI / screenWidth;
        cameraRotation += glm::vec2(xRot, yRot);

        int times2Pi = cameraRotation.x / (2 * M_PI);
        cameraRotation.x -= times2Pi * 2 * M_PI;
        if (cameraRotation.x < 0) cameraRotation.x += 2 * M_PI;

        if (0.5f * M_PI < cameraRotation.x && cameraRotation.x < 1.5f * M_PI) {
            cameraRotation.y -= 2 * yRot;
        }
    }

    lastMousePosition = glm::vec2(xpos, ypos);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    int vpWidth = 0.333 * screenWidth;
    int vpHeight = 0.9 * screenHeight;

    float x = (2.0f * ((int)lastMousePosition.x % vpWidth)) / vpWidth - 1.0f;
    float y = 1.0f - (2.0f * (lastMousePosition.y - 0.1f * screenHeight)) / vpHeight;
    float z = 1.0f;
    glm::vec3 ray_nds(x, y, z);

    glm::vec4 ray_clip(ray_nds.x, ray_nds.y, -1.0, 1.0);

    glm::vec4 ray_eye = inverse(projection) * ray_clip;

    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);

    glm::vec4 inv_ray_wor = inverse(model) * inverse(view) * ray_eye;

    auto rot = glm::mat4(1.0f);
    rot = glm::rotate(rot, cameraRotation.x, glm::vec3(1, 0, 0));
    rot = glm::rotate(rot, cameraRotation.y, glm::vec3(0, 1, 0));

    glm::vec3 ray_wor = rot * inv_ray_wor;
    ray_wor = normalize(ray_wor);

    cameraPosition += 0.05f * maxDim * (float)yoffset * ray_wor;
}

// glfw: whenever the window is resized, this callback is fired
void window_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
}
