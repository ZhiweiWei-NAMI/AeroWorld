// Copyright ChengWei, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/MeshComponent.h"
#include "ProceduralMeshComponent.h"
#include "CityBaseMeshComponent.generated.h"

USTRUCT(BlueprintType)
struct FCityMeshSection
{
	GENERATED_BODY();

	UPROPERTY()
	TArray<FVector> Vertices;

	TArray<int32> Triangles;

	TArray<FVector> Normals;

	TArray<FVector2D> UV0;

	void AddPosition(double* pos)
	{
		// cm -->m
		Vertices.Add(FVector(pos[0] - offset[0], pos[1] - offset[1], pos[2] - offset[2]) * 100);
	}
	void AddPosition(double pos0,double pos1,double pos2)
	{
		// cm -->m
		Vertices.Add(FVector(pos0 - offset[0], pos1 - offset[1], pos2 - offset[2]) * 100);
	}
	void AddUV(double a, double b)
	{
		UV0.Add(FVector2D(a, b));
	}
	void AddUV(double* uv)
	{
		UV0.Add(FVector2D(uv[0], uv[1]));
	}
	void AddNormal(double x, double y, double z)
	{
		Normals.Add(FVector(x, y, z));
	}
	int32 GetNumVertices()
	{
		return Vertices.Num();
	}
	int32 GetNumUV()
	{
		return UV0.Num();
	}
	int32 GetNumFaces()
	{
		return Triangles.Num() / 3;
	}
	void GetNormal(size_t index, double* xyz)
	{
		xyz[0] = Normals[index].X;
		xyz[1] = Normals[index].Y;
		xyz[2] = Normals[index].Z;
	}

	void GetUV(size_t index, double* xy)
	{
		xy[0] = UV0[index].X;
		xy[1] = UV0[index].Y;
	}
	void SetUV(size_t index, double* xyz)
	{
		UV0[index] = FVector2D(xyz[0], xyz[1]);
	}

	void ScaleUV(size_t start,size_t end,float scaleX,float scaleY)
	{
		FVector2D Factor(scaleX, scaleY);
		for(size_t i=start;i<end;i++)
		UV0[i]*= Factor;
	}
	void ScaleUV(size_t index, float scaleX, float scaleY)
	{
		FVector2D Factor(scaleX, scaleY);
		UV0[index] *= Factor;
	}
	void SetNormal(size_t index, double* xyz)
	{
		Normals[index] = FVector(xyz[0], xyz[1], xyz[2]);
	}

	void GetPosition(size_t index, double* xyz)
	{
		xyz[0] = Vertices[index].X;
		xyz[1] = Vertices[index].Y;
		xyz[2] = Vertices[index].Z;
	}
	void SetPosition(size_t index, double* xyz)
	{
		Vertices[index].X = xyz[0];
		Vertices[index].Y = xyz[1];
		Vertices[index].Z = xyz[2];
	}
	void AddTriangles(int32 x, int32 y, int32 z)
	{
		Triangles.Add(x);
		Triangles.Add(y);
		Triangles.Add(z);
	}
	double offset[3] = {0,0,0};
};
USTRUCT(BlueprintType)
struct FCityMeshSectionContext
{
GENERATED_BODY()
TArray<FCityMeshSection> Sections;
FString Name;
};
UCLASS(meta = (BlueprintSpawnableComponent), hidecategories = (Physics))
class CITYGENERATOR_API UCityBaseMeshComponent : public UProceduralMeshComponent
{
	GENERATED_BODY()
public:

	void Init(const int& SectionNum, double* _offset);

	void GenerateSection();

	void SetOffset(double* _offset) { std::memcpy(offset, _offset, 3 * sizeof(double)); }
public:
	TArray<FCityMeshSection> Sections;

	TArray<UMaterialInterface*> MaterialArray;
	double offset[3] = {};
};