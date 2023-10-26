#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Heat_method_3/Surface_mesh_geodesic_distances_3.h>
#include <CGAL/Surface_mesh_shortest_path.h>
#include <CGAL/boost/graph/dijkstra_shortest_paths.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "learnopengl/shader.h"
#include "learnopengl/camera.h"

#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
typedef CGAL::Polyhedron_3<K> Polyhedron;

typedef boost::graph_traits<SurfaceMesh>::vertex_descriptor VertexDescriptor;
typedef SurfaceMesh::Property_map<VertexDescriptor, double> VertexDistanceMap;
typedef CGAL::Heat_method_3::Surface_mesh_geodesic_distances_3<SurfaceMesh> HeatMethod;

typedef std::map<VertexDescriptor, int> VertexIndexMap;
typedef boost::associative_property_map<VertexIndexMap> VertexIdPropertyMap;

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

// dynamic arrays for geometry
GLfloat *vertices = NULL;
GLuint *edges = NULL;
GLfloat *coarseVertices = NULL;
GLuint *coarseEdges = NULL;
GLuint *candidateEdges = NULL;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 2.0f));

// mouse press callback
bool mousePressed;
bool firstCursorPositionCallbackOnPress;
glm::vec2 mousePressedPosition;
glm::vec2 currentMousePosition;

// rotation
glm::vec2 rotationAngles;
glm::vec2 rotationAnglesDrag;

// user settings
static float phi = 0.125f;
static float prevPhi = 0.0f;

CGAL::SM_Vertex_index selectedPoint = CGAL::SM_Vertex_index(0);
CGAL::SM_Vertex_index prevSelectedPoint = CGAL::SM_Vertex_index(0);
std::set<std::set<SurfaceMesh::Vertex_index>> coarseTriangles;


glm::vec3 selectedPointProjection;

bool useBarycentricCoords;
std::vector<SurfaceMesh::Vertex_index> barycentricPoints;
glm::vec3 barycentricCoords;

bool useEdgeCoords;
std::set<SurfaceMesh::Vertex_index> coordsEdge;
float edgeCoordsW1, edgeCoordsW2;

bool useInvDistWeights;
std::vector<SurfaceMesh::Vertex_index> threeClosestPoints;
glm::vec3 invDistWeights;

SurfaceMesh::Vertex_index coarsePoint;


glm::mat4 model, view, projection;
SurfaceMesh surface;
glm::vec3 rayCamPos;


std::vector<SurfaceMesh::Vertex_index> samplePoints(SurfaceMesh* surface) {
    float sumEdgeLengths = 0.0f;

    for (auto edgeIndex : surface->edges()) {
        auto h = surface->halfedge(edgeIndex);
        auto from = surface->point(surface->source(h));
        auto to = surface->point(surface->target(h));
        auto d = CGAL::sqrt(CGAL::squared_distance(from, to));
        sumEdgeLengths += (float) d;
    }

    float averageEdgeLength = sumEdgeLengths / surface->num_edges();
    float r = pow(phi, -1.0f/3.0f) * averageEdgeLength;

    std::vector<SurfaceMesh::Vertex_index> V;
    for (auto vertexIndex : surface->vertices()) {
        V.push_back(vertexIndex);
    }

    std::cout << V.size() << std::endl << std::endl;

    for (auto it = V.begin(); it < V.end(); it++) {
        VertexDistanceMap vertexDistance = surface->add_property_map<VertexDescriptor, double>("v:distance", 0).first;
        VertexDescriptor source = *it;
        HeatMethod hm(*surface);
        hm.add_source(source);
        hm.estimate_geodesic_distances(vertexDistance);

        auto withinRadius = [&vertexDistance, &r](auto v) { return get(vertexDistance, v) < r; };
        V.erase(std::remove_if(it + 1, V.end(), withinRadius), V.end());
    }

    std::cout << V.size() << std::endl << std::endl;
    return V;
}

std::vector<SurfaceMesh::Vertex_index> createNeighbourhoods(SurfaceMesh* surface, std::vector<SurfaceMesh::Vertex_index> sampling) {
    std::vector<std::pair<float, SurfaceMesh::Vertex_index>> closestSamplePoint(surface->num_vertices(), std::pair<float, SurfaceMesh::Vertex_index>(INFINITY, SurfaceMesh::Vertex_index(-1)));

    for (auto s : sampling) {
        // Associate indices to the vertices
        VertexIndexMap vertex_id_map;
        VertexIdPropertyMap vertex_index_pmap(vertex_id_map);
        int index = 0;
        for(VertexDescriptor vd : surface->vertices())
            vertex_id_map[vd] = index++;

        // Dijkstra's shortest path needs property maps for the predecessor and distance
        // We first declare a vector
        std::vector<VertexDescriptor> predecessor(surface->num_vertices());
        // and then turn it into a property map
        boost::iterator_property_map<std::vector<VertexDescriptor>::iterator, VertexIdPropertyMap>
                predecessor_pmap(predecessor.begin(), vertex_index_pmap);
        std::vector<double> distance(surface->num_vertices());
        boost::iterator_property_map<std::vector<double>::iterator, VertexIdPropertyMap>
                distance_pmap(distance.begin(), vertex_index_pmap);

        boost::dijkstra_shortest_paths(*surface, (VertexDescriptor)s,
                                       distance_map(distance_pmap)
                                               .predecessor_map(predecessor_pmap)
                                               .vertex_index_map(vertex_index_pmap));

        for (auto v : surface->vertices()) {
            float d = get(distance_pmap, v);
            if (d < closestSamplePoint.at(v.idx()).first) {
                closestSamplePoint.at(v.idx()) = std::pair<float, SurfaceMesh::Vertex_index>(d, s);
            }
        }
    }

    std::vector<SurfaceMesh::Vertex_index> res;
    std::transform(closestSamplePoint.begin(), closestSamplePoint.end(), std::back_inserter(res), [](auto p) { return p.second; });
    return res;
}

glm::vec3 toVec3(CGAL::Point_3<CGAL::Epick> p) {
    return {p.x(), p.y(), p.z()};
}

SurfaceMesh constructCoarserLevel(SurfaceMesh &surface, std::vector<SurfaceMesh::Vertex_index> &sampling, std::vector<SurfaceMesh::Vertex_index> &neighbourhoods) {
    SurfaceMesh res;

    for (auto v : sampling) {
        glm::vec3 neighbourhoodSumPoints(0);
        int neighbourhoodNPoints = 0;

        int i = 0;
        for (auto p : neighbourhoods) {
            if (p == v) {
                neighbourhoodSumPoints += toVec3(surface.point(SurfaceMesh::Vertex_index(i)));
                neighbourhoodNPoints++;
            }
            i++;
        }

        auto mean = neighbourhoodSumPoints / (float)neighbourhoodNPoints;
        res.add_vertex(CGAL::Point_3<CGAL::Epick>(mean.x, mean.y, mean.z));
    }

    std::vector<std::set<SurfaceMesh::Vertex_index>> neighboursList(res.num_vertices());

    for (auto e : surface.edges()) {
        // if endpoints of e are in different neighbourhoods, then create edge in res between those neighbourhoods
        auto from = surface.source(e.halfedge());
        auto to = surface.target(e.halfedge());

        auto neighbourhoodFrom = neighbourhoods.at(from.idx());
        auto neighbourhoodTo = neighbourhoods.at(to.idx());

        if (neighbourhoodFrom != neighbourhoodTo) {
            auto newFromIdx = std::distance(sampling.begin(), std::find(sampling.begin(), sampling.end(), neighbourhoodFrom));
            auto newToIdx = std::distance(sampling.begin(), std::find(sampling.begin(), sampling.end(), neighbourhoodTo));
            res.add_edge(*(res.vertices_begin() + newFromIdx), *(res.vertices_begin() + newToIdx));

            neighboursList.at(newFromIdx).insert((SurfaceMesh::Vertex_index)newToIdx);
            neighboursList.at(newToIdx).insert((SurfaceMesh::Vertex_index)newFromIdx);
        }
    }

    coarseTriangles.clear();

    for (auto v1 : res.vertices()) {
        for (auto v2 : neighboursList.at(v1)) {
            for (auto v3 : neighboursList.at(v2)) {
                if (std::find(neighboursList.at(v3).begin(), neighboursList.at(v3).end(), v1) != neighboursList.at(v3).end()) {
                    std::set<SurfaceMesh::Vertex_index> triangle;
                    triangle.insert(v1);
                    triangle.insert(v2);
                    triangle.insert(v3);
                    coarseTriangles.insert(triangle);
                }
            }
        }
    }

    return res;
}

float inTriangle(SurfaceMesh mesh, glm::vec3 p, std::vector<CGAL::SM_Vertex_index> tri, glm::vec3& pProjected, glm::vec3& bary, std::map<CGAL::SM_Vertex_index, float>& insideEdge) {
    auto v1 = toVec3(mesh.point(tri.at(0)));
    auto v2 = toVec3(mesh.point(tri.at(1)));
    auto v3 = toVec3(mesh.point(tri.at(2)));
    auto v1ToP = p - v1;
    auto e12 = v2 - v1;
    auto e13 = v3 - v1;
    auto triNormal = glm::normalize(glm::cross(e12, e13));

    float distToTriangle = glm::dot(v1ToP, triNormal);
    pProjected = p - distToTriangle * triNormal;

    float doubleArea = glm::dot(glm::cross(e12, e13), triNormal);
    bary.x = glm::dot(glm::cross(v3 - v2, pProjected - v2), triNormal) / doubleArea;
    bary.y = glm::dot(glm::cross(v1 - v3, pProjected - v3), triNormal) / doubleArea;
    bary.z = 1.0f - bary.x - bary.y;

    if (insideEdge.find(tri.at(1)) == insideEdge.end()) {
        insideEdge[tri.at(1)] = glm::length(v1ToP - glm::dot(v1ToP, e12) * e12);
    }
    if (insideEdge.find(tri.at(2)) == insideEdge.end()) {
        insideEdge[tri.at(2)] = glm::length(v1ToP - glm::dot(v1ToP, e13) * e13);
    }
    if (bary.x < 0.0f || bary.y < 0.0f) {
        insideEdge[tri.at(1)] = -1.0f;
    }
    if (bary.x < 0.0f || bary.z < 0.0f) {
        insideEdge[tri.at(2)] = -1.0f;
    }

    if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
        return abs(distToTriangle);
    }

    return -1.0f;
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

    // build and compile our shader program
    // ------------------------------------
    Shader ourShader("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");

    // load mesh from file
    // -------------------
    Polyhedron mesh;

    std::ifstream in("meshes/cactus.off");
    in >> mesh;
    CGAL::copy_face_graph( mesh, surface);

    auto nVertices = surface.num_vertices();
    auto nEdges = surface.num_edges();

    std::cout << "#vertices = " << nVertices << std::endl;
    std::cout << "#edges = " << nEdges << std::endl;

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    vertices = new float[nVertices * 3];
    edges = new uint[nEdges * 2];

    std::vector<glm::vec3> vertexColorsLeft(nVertices);
    std::vector<glm::vec3> vertexColorsRight(nVertices);
    std::vector<glm::vec3> coarseVertexColors;

    auto contains = [](auto v, auto x) { return std::find(v.begin(), v.end(), x) != v.end(); };

    for (auto vertexIndex : surface.vertices()) {
        auto i = vertexIndex.idx();
        auto p = surface.point(vertexIndex);

        vertices[i*3]     = (float) p.x();
        vertices[i*3 + 1] = (float) p.y();
        vertices[i*3 + 2] = (float) p.z();
    }

    for (auto edgeIndex : surface.edges()) {
        auto i = edgeIndex.idx();
        auto h = surface.halfedge(edgeIndex);

        edges[i * 2]     = surface.source(h).idx();
        edges[i * 2 + 1] = surface.target(h).idx();
    }

    unsigned int vertexArray, vertexBuffer, colorBuffer, edgeBuffer, coarseVertexArray, coarseVertexBuffer, coarseColorBuffer, coarseEdgeBuffer, coarseCandidateEdgeBuffer, finalTriangleBuffer;
    glGenVertexArrays(1, &vertexArray);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &colorBuffer);
    glGenBuffers(1, &edgeBuffer);
    glGenVertexArrays(1, &coarseVertexArray);
    glGenBuffers(1, &coarseVertexBuffer);
    glGenBuffers(1, &coarseColorBuffer);
    glGenBuffers(1, &coarseEdgeBuffer);
    glGenBuffers(1, &coarseCandidateEdgeBuffer);
    glGenBuffers(1, &finalTriangleBuffer);

    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(vertexArray);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, 3 * nVertices * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    // position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    // color attribute
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * nEdges * sizeof(GLuint), edges, GL_STATIC_DRAW);

    std::vector<CGAL::SM_Vertex_index> sampling;
    std::vector<CGAL::SM_Vertex_index> neighbourhoods;
    SurfaceMesh coarserLevel;
    std::set<std::set<CGAL::SM_Vertex_index>> triEdges;

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
        ImGui::SetWindowSize(ImVec2(120, 40));

        ImGui::DragFloat("phi", &phi, 0.005f);
        if (!ImGui::IsItemActive() && (phi != prevPhi || selectedPoint != prevSelectedPoint)) {
            prevPhi = phi;
            prevSelectedPoint = selectedPoint;

            sampling = samplePoints(&surface);
            neighbourhoods = createNeighbourhoods(&surface, sampling);
            coarserLevel = constructCoarserLevel(surface, sampling, neighbourhoods);

            // For selected fine point
            auto closestCoarsePoint = neighbourhoods.at(selectedPoint);
            auto coarserLevelIdx = (SurfaceMesh::Vertex_index) std::distance(sampling.begin(), std::find(sampling.begin(), sampling.end(), closestCoarsePoint));
            coarsePoint = coarserLevelIdx;
            std::vector<std::set<SurfaceMesh::Vertex_index>> candidateTriangles;
            std::copy_if(coarseTriangles.begin(), coarseTriangles.end(), std::back_inserter(candidateTriangles), [coarserLevelIdx](auto t) { return std::find(t.begin(), t.end(), coarserLevelIdx) != t.end(); });
            triEdges = std::accumulate(candidateTriangles.begin(), candidateTriangles.end(), std::set<std::set<SurfaceMesh::Vertex_index>>(), [](auto acc, auto t) {
                std::set<SurfaceMesh::Vertex_index> e0, e1, e2;
                auto it = t.begin();
                e0.insert(*it);
                e1.insert(*it);
                it++;
                e1.insert(*it);
                e2.insert(*it);
                it++;
                e2.insert(*it);
                e0.insert(*it);
                acc.insert(e0);
                acc.insert(e1);
                acc.insert(e2);
                return acc;
            });

            float minDist = INFINITY;
            std::vector<SurfaceMesh::Vertex_index> minTriangle;
            bool triangleFound = false;
            std::map<CGAL::SM_Vertex_index, float> insideEdge;

            for (auto t : candidateTriangles) {
                auto triangle = std::vector<SurfaceMesh::Vertex_index>(t.begin(), t.end());
                while (triangle.at(0) != coarserLevelIdx) {
                    std::rotate(triangle.begin(), triangle.begin() + 1, triangle.end());
                }
                glm::vec3 pProjected, bary;
                auto dist = inTriangle(coarserLevel, toVec3(surface.point(selectedPoint)), triangle, pProjected, bary, insideEdge);

                if (dist >= 0.0f && dist < minDist) {
                    triangleFound = true;
                    minDist = dist;
                    minTriangle = triangle;
                    selectedPointProjection = pProjected;
                    barycentricCoords = bary;
                }
            }

            useEdgeCoords = false;
            useInvDistWeights = false;
            if (triangleFound) {
                // then we can just use the bary coords
                std::cout << "barycentric triangle coords" << std::endl;

                useBarycentricCoords = true;
                barycentricPoints = minTriangle;
            } else {
                useBarycentricCoords = false;

                bool edgeFound = false;
                float minEdgeDist = INFINITY;
                CGAL::SM_Vertex_index minEdge;

                for (auto element : insideEdge) {
                    if (element.second >= 0.0f && element.second < minEdgeDist) {
                        edgeFound = true;
                        minEdgeDist = element.second;
                        minEdge = element.first;
                    }
                }

                if (edgeFound) {
                    std::cout << "barycentric edge coords" << std::endl;
                    useEdgeCoords = true;
                    // then we can use "bary" coords for edge
                    auto finePoint = toVec3(surface.point(selectedPoint));
                    auto coarsePoint = toVec3(coarserLevel.point(coarserLevelIdx));
                    auto p2 = toVec3(coarserLevel.point(minEdge));
                    auto e12 = p2 - coarsePoint;
                    float e12Length = std::max(glm::length(e12), 1e-8f);
                    float w2 = glm::dot(finePoint - coarsePoint, glm::normalize(e12)) / e12Length;
                    w2 = std::min(std::max(w2, 0.0f), 1.0f);
                    float w1 = 1.0f - w2;
                    // w1, w2 are "bary" coords
                    coordsEdge = { coarserLevelIdx, minEdge };
                    edgeCoordsW1 = w1;
                    edgeCoordsW2 = w2;
                    selectedPointProjection = w1 * coarsePoint + w2 * p2;
                } else {
                    // Use closest three
                    std::cout << "closest three" << std::endl;
                    useInvDistWeights = true;
                    std::vector<SurfaceMesh::Vertex_index> prolongFrom(3);
                    prolongFrom[0] = coarserLevelIdx;

                    std::vector<std::pair<SurfaceMesh::Vertex_index, float>> pointsDistances;
                    for (auto e : coarserLevel.edges()) {
                        auto h = coarserLevel.halfedge(e);
                        SurfaceMesh::Vertex_index neighbourIdx;

                        if (coarserLevel.source(h) == coarserLevelIdx) {
                            neighbourIdx = coarserLevel.target(h);
                        } else if (coarserLevel.target(h) == coarserLevelIdx) {
                            neighbourIdx = coarserLevel.source(h);
                        } else {
                            continue;
                        }

                        if (std::find_if(pointsDistances.begin(), pointsDistances.end(), [&neighbourIdx](auto pair) { return pair.first == neighbourIdx; }) != pointsDistances.end()) {
                            continue;
                        }

                        float dist = glm::distance(toVec3(surface.point(selectedPoint)), toVec3(coarserLevel.point(neighbourIdx)));
                        pointsDistances.push_back({ neighbourIdx, dist });
                    }

                    std::sort(pointsDistances.begin(), pointsDistances.end(), [](auto lhs, auto rhs) { return rhs.second - lhs.second; });
                    prolongFrom[1] = pointsDistances.at(0).first;
                    prolongFrom[2] = pointsDistances.at(1).first;

                    // Compute inverse distance weights
                    double sumWeight = 0.;
                    std::vector<double> weights(3);
                    for (int j = 0; j < 3; ++j) {
                        float dist = glm::distance(toVec3(surface.point(selectedPoint)), toVec3(coarserLevel.point(prolongFrom.at(j))));
                        weights[j] = 1. / std::max(1e-8f, dist);
                        sumWeight += weights[j];
                    }
                    for (int j = 0; j < weights.size(); ++j) {
                        weights[j] = weights[j] / sumWeight;
                    }

                    threeClosestPoints = prolongFrom;
                    invDistWeights = glm::vec3(weights[0], weights[1], weights[2]);
                }
            }
            std::cout << "useBarycentricCoords = " << useBarycentricCoords << std::endl;

            for (auto vertexIndex : surface.vertices()) {
                auto i = vertexIndex.idx();
                vertexColorsLeft.at(i) = contains(sampling, vertexIndex) ? glm::vec3(1, 1, 0) : glm::vec3(0);

                if (i == selectedPoint) {
                    vertexColorsLeft.at(i) = glm::vec3(1.0f, 0.06f, 0.94f);
                }

                int neighbourhood = neighbourhoods.at(i).idx();
                vertexColorsRight.at(i) = glm::vec3(((neighbourhood * 3) % 255) / 255.0f, ((neighbourhood * 5) % 255) / 255.0f, ((neighbourhood * 7) % 255) / 255.0f);
            }

            // set up vertex data (and buffer(s)) and configure vertex attributes
            // ------------------------------------------------------------------
            coarseVertices = new float[(coarserLevel.num_vertices() + 1) * 3];
            coarseEdges = new uint[(coarserLevel.num_edges() + 3) * 2];
            candidateEdges = new uint[triEdges.size() * 2];

            coarseVertexColors = std::vector<glm::vec3>(coarserLevel.num_vertices());

            for (auto vertexIndex : coarserLevel.vertices()) {
                auto i = vertexIndex.idx();
                auto p = coarserLevel.point(vertexIndex);

                coarseVertices[i*3]     = (float) p.x();
                coarseVertices[i*3 + 1] = (float) p.y();
                coarseVertices[i*3 + 2] = (float) p.z();

//                coarseVertexColors.at(i) = glm::vec3((((int)pow(i, 3)) % 255) / 255.0f, (((int)pow(i, 4)) % 255) / 255.0f, (((int)pow(i, 5)) % 255) / 255.0f);
            }

            if (useBarycentricCoords || useEdgeCoords) {
                coarseVertices[coarserLevel.num_vertices() * 3]     = selectedPointProjection.x;
                coarseVertices[coarserLevel.num_vertices() * 3 + 1] = selectedPointProjection.y;
                coarseVertices[coarserLevel.num_vertices() * 3 + 2] = selectedPointProjection.z;
            } else if (useInvDistWeights) {
                auto p = surface.point(selectedPoint);
                coarseVertices[coarserLevel.num_vertices() * 3]     = p.x();
                coarseVertices[coarserLevel.num_vertices() * 3 + 1] = p.y();
                coarseVertices[coarserLevel.num_vertices() * 3 + 2] = p.z();
            }

            for (auto edgeIndex : coarserLevel.edges()) {
                auto i = edgeIndex.idx();
                auto h = coarserLevel.halfedge(edgeIndex);

                coarseEdges[i * 2]     = coarserLevel.source(h).idx();
                coarseEdges[i * 2 + 1] = coarserLevel.target(h).idx();
            }

            if (useBarycentricCoords) {
                for (int i = 0; i < 3; i++) {
                    coarseEdges[(coarserLevel.num_edges() + i) * 2]     = coarserLevel.num_vertices(); // selected point projection
                    coarseEdges[(coarserLevel.num_edges() + i) * 2 + 1] = barycentricPoints.at(i);
                }
            } else if (useInvDistWeights) {
                for (int i = 0; i < 3; i++) {
                    coarseEdges[(coarserLevel.num_edges() + i) * 2]     = coarserLevel.num_vertices(); // selected point
                    coarseEdges[(coarserLevel.num_edges() + i) * 2 + 1] = threeClosestPoints.at(i);
                }
            }

            int candidateEdgeIndex = 0;
            for (auto e : triEdges) {
                auto i = candidateEdgeIndex++;

                auto it = e.begin();
                candidateEdges[i * 2]     = (*it++).idx();
                candidateEdges[i * 2 + 1] = (*it).idx();
            }

            // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
            glBindVertexArray(coarseVertexArray);

            glBindBuffer(GL_ARRAY_BUFFER, coarseVertexBuffer);
            glBufferData(GL_ARRAY_BUFFER, 3 * (coarserLevel.num_vertices() + 1) * sizeof(GLfloat), coarseVertices, GL_STATIC_DRAW);

            glBindBuffer(GL_ARRAY_BUFFER, coarseColorBuffer);
            glBufferData(GL_ARRAY_BUFFER, coarseVertexColors.size() * sizeof(glm::vec3), &coarseVertexColors[0], GL_STATIC_DRAW);

            // position attribute
            glBindBuffer(GL_ARRAY_BUFFER, coarseVertexBuffer);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(0);

            // color attribute
            glBindBuffer(GL_ARRAY_BUFFER, coarseColorBuffer);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(1);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseEdgeBuffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * (coarserLevel.num_edges() + 3) * sizeof(GLuint), coarseEdges, GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseCandidateEdgeBuffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * triEdges.size() * sizeof(GLuint), candidateEdges, GL_STATIC_DRAW);

            std::vector<int> finalTriangle;
            std::transform(barycentricPoints.begin(), barycentricPoints.end(), std::back_inserter(finalTriangle), [](auto p) { return p.idx(); });

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, finalTriangleBuffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(GLuint), &finalTriangle[0], GL_STATIC_DRAW);
        }

        ImGui::End();

        ImGui::Begin("Prolongation", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration);
        ImGui::SetWindowPos(ImVec2(screenWidth - 300 - 10, 10));
        ImGui::SetWindowSize(ImVec2(300, 120));

        if (useBarycentricCoords) {
            ImGui::Text("Prolongation:");
            ImGui::Text("Barycentric coordinates in triangle");
            ImGui::Dummy(ImVec2(0.0f, 8.0f));
            ImGui::Text("v1: %f", barycentricCoords.x);
            ImGui::Text("v2: %f", barycentricCoords.y);
            ImGui::Text("v3: %f", barycentricCoords.z);
        } else if (useEdgeCoords) {
            ImGui::Text("Prolongation:");
            ImGui::Text("Barycentric coordinates on edge");
            ImGui::Dummy(ImVec2(0.0f, 8.0f));
            ImGui::Text("v1: %f", edgeCoordsW1);
            ImGui::Text("v2: %f", edgeCoordsW2);
        } else {
            ImGui::Text("Prolongation:");
            ImGui::Text("Inverse distance weights");
            ImGui::Dummy(ImVec2(0.0f, 8.0f));
            ImGui::Text("v1: %f", invDistWeights.x);
            ImGui::Text("v2: %f", invDistWeights.y);
            ImGui::Text("v3: %f", invDistWeights.z);
        }

        ImGui::End();

        // input
        // -----
        processInput(window);

        int vpWidth = 0.333 * frameBufferWidth;
        int vpHeight = 0.9 * frameBufferHeight;

        // pass projection matrix to shader (note that in this case it could change every frame)
        projection = glm::perspective(glm::radians(camera.Zoom), (float)vpWidth / (float)vpHeight, 0.1f, 100.0f);
        ourShader.setMat4("projection", projection);

        // camera/view transformation
        view = camera.GetViewMatrix();
        ourShader.setMat4("view", view);

        // calculate the model matrix and pass it to shader before drawing
        model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        model = glm::rotate(model, rotationAngles.x + rotationAnglesDrag.x, glm::vec3(1, 0, 0));
        model = glm::rotate(model, rotationAngles.y + rotationAnglesDrag.y, glm::vec3(0, 1, 0));
        ourShader.setMat4("model", model);

        // set point size
        glPointSize(5 * 45 / camera.Zoom);

        // render
        // ------
        glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0, 0, vpWidth, vpHeight);

        // render the triangle
        ourShader.use();

        glBindVertexArray(vertexArray);

        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
        glBufferData(GL_ARRAY_BUFFER, vertexColorsLeft.size() * sizeof(glm::vec3), &vertexColorsLeft[0], GL_STATIC_DRAW);

        // draw edges
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
        glDrawElements(GL_LINES, 2 * nEdges, GL_UNSIGNED_INT, 0);

        // draw vertices
        glEnableVertexAttribArray(1);
        glBindVertexArray(vertexArray);
        glDrawArrays(GL_POINTS, 0, nVertices);

        glViewport(vpWidth, 0, vpWidth, vpHeight);

        // render the triangle
        ourShader.use();

        glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
        glBufferData(GL_ARRAY_BUFFER, vertexColorsRight.size() * sizeof(glm::vec3), &vertexColorsRight[0], GL_STATIC_DRAW);

        // draw edges
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeBuffer);
        glDrawElements(GL_LINES, 2 * nEdges, GL_UNSIGNED_INT, 0);

        // draw vertices
        glEnableVertexAttribArray(1);
        glBindVertexArray(vertexArray);
        glDrawArrays(GL_POINTS, 0, nVertices);

        glViewport(2 * vpWidth, 0, vpWidth, vpHeight);

        // render the triangle
        ourShader.use();

        glBindVertexArray(coarseVertexArray);

        // draw edges
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseEdgeBuffer);
        glDrawElements(GL_LINES, 2 * coarserLevel.num_edges(), GL_UNSIGNED_INT, 0);

        // draw candidate edges
        ourShader.setBool("useUniformColor", true);
        ourShader.setVec3("uColor", glm::vec3(1, 1, 0));
        glDisableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseCandidateEdgeBuffer);
        glDrawElements(GL_LINES, 2 * triEdges.size(), GL_UNSIGNED_INT, 0);

        if (useBarycentricCoords) {
            ourShader.setVec3("uColor", glm::vec3(0, 0.8f, 1));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, finalTriangleBuffer);
            glDrawElements(GL_LINE_LOOP, 3, GL_UNSIGNED_INT, 0);
        } else if (useEdgeCoords) {
            int coordsEdgeIndex = -1;
            for (auto edgeIndex : coarserLevel.edges()) {
                auto i = edgeIndex.idx();
                auto h = coarserLevel.halfedge(edgeIndex);

                if (std::set<SurfaceMesh::Vertex_index>({ coarserLevel.source(h), coarserLevel.target(h) }) == coordsEdge) {
                    coordsEdgeIndex = edgeIndex;
                    break;
                }
            }

            ourShader.setVec3("uColor", glm::vec3(1, 0, 0));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseEdgeBuffer);
            glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT, (void*) (coordsEdgeIndex * 2 * sizeof(GLuint)));
        }

        if (useBarycentricCoords || useInvDistWeights) {
            ourShader.setVec3("uColor", glm::vec3(1, 0, 0));
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coarseEdgeBuffer);
            glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, (void*) (coarserLevel.num_edges() * 2 * sizeof(GLuint)));
        }

        if (useBarycentricCoords || useEdgeCoords) {
            // draw projected point
            ourShader.setVec3("uColor", glm::vec3(0.6f, 0, 0.8f));
            glBindVertexArray(coarseVertexArray);
            glDrawArrays(GL_POINTS, coarserLevel.num_vertices(), 1);

            // draw coarse point
            ourShader.setVec3("uColor", glm::vec3(1, 1, 0));
            glBindVertexArray(coarseVertexArray);
            glDrawArrays(GL_POINTS, coarsePoint.idx(), 1);
        }

        // draw selected point
        ourShader.setVec3("uColor", glm::vec3(1, 0, 1));
        glBindVertexArray(vertexArray);
        glDrawArrays(GL_POINTS, selectedPoint.idx(), 1);

        // draw vertices
        ourShader.setBool("useUniformColor", false);
        glEnableVertexAttribArray(1);
        glBindVertexArray(coarseVertexArray);
//        glDrawArrays(GL_POINTS, 0, coarserLevel.num_vertices());

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
    glDeleteBuffers(1, &colorBuffer);
    glDeleteBuffers(1, &edgeBuffer);
    glDeleteBuffers(1, &coarseVertexArray);
    glDeleteBuffers(1, &coarseVertexBuffer);
    glDeleteBuffers(1, &coarseColorBuffer);
    glDeleteBuffers(1, &coarseEdgeBuffer);
    glDeleteBuffers(1, &coarseCandidateEdgeBuffer);
    glDeleteBuffers(1, &finalTriangleBuffer);

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
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            firstCursorPositionCallbackOnPress = true;
        } else {
            mousePressed = false;
            firstCursorPositionCallbackOnPress = false;

            if (rotationAnglesDrag == glm::vec2(0)) {
                int vpWidth = 0.333 * screenWidth;
                int vpHeight = 0.9 * screenHeight;

                if (currentMousePosition.x < vpWidth && currentMousePosition.y > 0.1 * screenHeight) {
                    float x = (2.0f * ((int)currentMousePosition.x % vpWidth)) / vpWidth - 1.0f;
                    float y = 1.0f - (2.0f * (currentMousePosition.y - 0.1f * screenHeight)) / vpHeight;
                    float z = 1.0f;
                    glm::vec3 ray_nds(x, y, z);

                    glm::vec4 ray_clip(ray_nds.x, ray_nds.y, -1.0, 1.0);

                    glm::vec4 ray_eye = inverse(projection) * ray_clip;

                    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);

                    glm::vec4 inv_ray_wor = inverse(model) * inverse(view) * ray_eye;
                    glm::vec3 ray_wor = glm::vec3(inv_ray_wor.x, inv_ray_wor.y, inv_ray_wor.z);
                    ray_wor = normalize(ray_wor);

                    rayCamPos = inverse(model) * glm::vec4(camera.Position.x, camera.Position.y, camera.Position.z, 1.0f);

                    float tMin = INFINITY;
                    SurfaceMesh::Vertex_index vMin;
                    bool intersection = false;
                    int intersections = 0;

                    float radius2 = 0.0001f;

                    for (auto v : surface.vertices()) {
                        float t0, t1; // solutions for t if the ray intersects

                        // geometric solution
                        auto vPos = toVec3(surface.point(v));
                        glm::vec3 L = vPos - rayCamPos;
                        float tca = glm::dot(L, ray_wor);
                        // if (tca < 0) return false;
                        float d2 = glm::dot(L, L) - tca * tca;
                        if (d2 > radius2) continue;
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

            rotationAngles += rotationAnglesDrag;
            rotationAnglesDrag = glm::vec2(0);
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
        if (firstCursorPositionCallbackOnPress) {
            mousePressedPosition = glm::vec2(xpos, ypos);
            firstCursorPositionCallbackOnPress = false;
        }

        float xRotationAngleDrag = (ypos - mousePressedPosition.y) * 2 * M_PI / screenHeight;
        float yRotationAngleDrag = (xpos - mousePressedPosition.x) * 2 * M_PI / screenWidth;
        rotationAnglesDrag = glm::vec2(xRotationAngleDrag, yRotationAngleDrag);
    }

    currentMousePosition = glm::vec2(xpos, ypos);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

// glfw: whenever the window is resized, this callback is fired
void window_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
}
