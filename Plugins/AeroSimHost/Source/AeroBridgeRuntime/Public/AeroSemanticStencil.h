#pragma once

#include "CoreMinimal.h"

class FJsonObject;
class AActor;
class UPrimitiveComponent;
class UWorld;

struct FAeroSemanticStencilRule
{
	FString ClassName;
	uint8 ClassId = 0;
	int32 Priority = 0;
	int32 Order = 0;
	TArray<FString> Patterns;
	TArray<FString> ActorPatterns;
	TArray<FString> ComponentPatterns;
	TArray<FString> ActorClassPatterns;
	TArray<FString> ComponentClassPatterns;
	TArray<FString> TagPatterns;
	TArray<FString> MaterialPatterns;
	TArray<FString> FolderPatterns;
};

struct FAeroSemanticStencilComponentAudit
{
	FString ActorName;
	FString ActorLabel;
	FString ActorClass;
	FString ComponentName;
	FString ComponentClass;
	FString FolderPath;
	TArray<FString> Tags;
	TArray<FString> Materials;
	bool bVisible = false;
	bool bRegistered = false;
	bool bRenderCustomDepthBefore = false;
	int32 StencilBefore = 0;
	int32 StencilAfter = 0;
	uint8 MatchedClassId = 0;
	FString MatchedClassName = TEXT("ignore");
	FString MatchedRulePattern;
	FString Bounds;
};

struct FAeroSemanticStencilAudit
{
	FString RulesPath;
	FString CaptureMaterialPath;
	FString CaptureEncoding;
	bool bAssigned = false;
	int32 ActorCount = 0;
	int32 PrimitiveComponentCount = 0;
	int32 VisiblePrimitiveComponentCount = 0;
	int32 RegisteredPrimitiveComponentCount = 0;
	int32 AssignedComponentCount = 0;
	TMap<FString, uint8> ClassNameToId;
	TMap<uint8, FString> ClassIdToName;
	TMap<uint8, int32> MatchedComponentHistogram;
	TMap<uint8, int32> AssignedComponentHistogram;
	TArray<FAeroSemanticStencilComponentAudit> Components;
};

namespace AeroSemanticStencil
{
FString DefaultRulesPath();

bool LoadRules(
	const FString& RulesPath,
	TArray<FAeroSemanticStencilRule>& OutRules,
	TMap<FString, uint8>& OutClassNameToId,
	TMap<uint8, FString>& OutClassIdToName,
	FString& OutError,
	FString* OutCaptureMaterialPath = nullptr,
	FString* OutCaptureEncoding = nullptr);

bool AuditAndAssign(
	UWorld* World,
	const FString& RulesPath,
	bool bAssign,
	const TSet<const AActor*>& IgnoredActors,
	FAeroSemanticStencilAudit& OutAudit,
	FString& OutError);

TSharedPtr<FJsonObject> AuditToJson(const FAeroSemanticStencilAudit& Audit, bool bIncludeComponents);

bool SaveAuditJson(const FAeroSemanticStencilAudit& Audit, const FString& AuditPath, FString& OutError);
}
